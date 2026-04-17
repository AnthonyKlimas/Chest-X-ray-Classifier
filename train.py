# train_swin.py
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import timm

# Import data module
from dataset import (
    CLIP_LIMIT, TILE_GRID_SIZE, CXR8Dataset, CLAHETransform, PerImageStandardize, make_train_tf,
    worker_init_fn, ALL_CLASSES
)

# Paths
METADATA_CSV_PATH = "../chest_xray_dataset/CXR8/Data_Entry_2017_v2020.csv"
IMAGE_ROOT = r"C:\Users\nick\computing_for_health_and_medicine\chest_xray_dataset\CXR8\images"
MODEL_OUTPUT_FILE = "swin_cxr8_best.pth"

# Hyperparameters
SAMPLER_POWER = 0.25
WARMUP_EPOCHS = 3
WARMUP_START_FACTOR = 0.3
WARMUP_END_FACTOR = 1.0
NUM_EPOCHS = 22
BASE_LR = 1e-3
PATIENCE  = 4
WEIGHT_DECAY = 1e-2
BATCH_SIZE_VAL = 16
BATCH_SIZE_TRAIN = 32

NO_FINDING_COL = ALL_CLASSES.index("No Finding")
# RARE_CLASSES = ["Hernia", "Fibrosis", "Pneumonia"]
# RARE_COLS = [ALL_gCLASSES.index(c) for c in RARE_CLASSES]
NUM_CLASSES = len(ALL_CLASSES)


# Loss
class AsymmetricLoss(torch.nn.Module):
    def __init__(self, gamma_neg=2, gamma_pos=0, clip=0.05, eps=1e-6):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits.clamp(-10, 10))

        loss_pos = targets * torch.log(probs.clamp(min=1e-6))
        
        probs_neg = (1 - probs + self.clip).clamp(max=1.0)
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=1e-6))
        
        loss = loss_pos * (1 - probs) ** self.gamma_pos + \
               loss_neg * probs ** self.gamma_neg

        return -loss.mean()



# Model Wrapper
class SwinWithView(torch.nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        C = backbone.head.in_features
        backbone.head = torch.nn.Identity()
        self.backbone = backbone

        self.view_embed = torch.nn.Embedding(2, 32)
        self.view_proj  = torch.nn.Linear(32, C)
        self.view_scale = torch.nn.Parameter(torch.tensor(0.3))

        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(C),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(C, 512),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, num_classes),
        )

    def forward(self, x, view_id):
        feats = self.backbone.forward_features(x)
        if feats.ndim == 4:
            feats = feats.mean(dim=(1, 2))

        v = self.view_embed(view_id)
        v = self.view_proj(v)

        feats = feats + self.view_scale * v
        return self.head(feats)


# Param Groups
def build_param_groups(model, base_lr=1e-4, decay=0.8):
    layers = [model.backbone.layers[i] for i in range(len(model.backbone.layers))]
    groups = []
    for i, layer in enumerate(reversed(layers)):
        lr = base_lr * (decay ** i)
        groups.append({"params": layer.parameters(), "lr": lr})

    groups.append({"params": model.head.parameters(), "lr": base_lr})
    groups.append({"params": list(model.view_embed.parameters()) +
                              list(model.view_proj.parameters()), "lr": base_lr})

    registered = set(id(p) for g in groups for p in g["params"])
    others = [p for p in model.parameters() if id(p) not in registered]
    if others:
        groups.append({"params": others, "lr": base_lr * (decay ** len(layers))})

    return groups


# Main
if __name__ == "__main__":

    # GPU prep
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Load metadata
    df = pd.read_csv(METADATA_CSV_PATH)
    df = df[["Image Index", "Finding Labels", "View Position"]].copy()
    df["view_id"] = df["View Position"].map({"PA": 0, "AP": 1}).fillna(0).astype(int)
    df = df[df["View Position"].isin(["PA", "AP"])].reset_index(drop=True)
    df["labels"] = df["Finding Labels"].str.split("|")

    # Image lookup
    all_png = glob.glob(os.path.join(IMAGE_ROOT, "**", "*.png"), recursive=True)
    path_lookup = {os.path.basename(p): p for p in all_png}

    # Labels
    mlb = MultiLabelBinarizer(classes=ALL_CLASSES)
    label_matrix = mlb.fit_transform(df["labels"])
    mask = df["Image Index"].isin(path_lookup)
    df = df[mask].reset_index(drop=True)
    label_matrix = label_matrix[mask.values]

    # Split
    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=42)


    # Transforms
    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        CLAHETransform(clip_limit=CLIP_LIMIT, tile_grid_size=(TILE_GRID_SIZE, TILE_GRID_SIZE)),
        transforms.ToTensor(),
        PerImageStandardize()
    ])

    train_tf = make_train_tf(256)
    
    train_ds = CXR8Dataset(df, label_matrix, train_idx, train_tf, path_lookup)
    val_ds   = CXR8Dataset(df, label_matrix, val_idx,   val_tf,   path_lookup)
    # After building train_ds, verify alignment:
    img, lbl, view = train_ds[0]
    expected_label = label_matrix[train_idx[0]]
    assert np.array_equal(lbl.numpy(), expected_label), "Label mismatch!"

    # Sampler
    class_counts = label_matrix[train_idx].sum(axis=0)
    class_weights = 1.0 / (class_counts + 1e-6) ** SAMPLER_POWER
    sample_weights = (label_matrix[train_idx] * class_weights).sum(axis=1)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_TRAIN,
        sampler=sampler,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        persistent_workers=False,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        persistent_workers=False,
        pin_memory=True,
    )

    # Model
    base = timm.create_model(
        "swinv2_small_window8_256",
        img_size=256,
        pretrained=True,
        num_classes=0,
    )
    model = SwinWithView(base, NUM_CLASSES).to(device)

    param_group = build_param_groups(model, base_lr=BASE_LR, decay=0.8)
    scaler = GradScaler(device="cuda")
    criterion = AsymmetricLoss()
    optimizer = torch.optim.AdamW(param_group, weight_decay=WEIGHT_DECAY)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=WARMUP_START_FACTOR,
        end_factor=WARMUP_END_FACTOR,
        total_iters=WARMUP_EPOCHS
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS - WARMUP_EPOCHS,
        eta_min=1e-6
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS]
    )

    # Training Loop
    def run_epoch(loader, train=True, scaler=None):
        model.train() if train else model.eval()
        total_loss = 0.0
        all_logits, all_labels = [], []

        with torch.set_grad_enabled(train):
            for imgs, lbls, views in tqdm(loader, desc="train" if train else "val ", leave=False):
                imgs  = imgs.to(device)
                lbls  = lbls.to(device)
                views = views.to(device)

                if train:
                    original_lbls = lbls.clone()
                    optimizer.zero_grad()

                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    logits = model(imgs, views)

                loss = criterion(logits.float(), lbls.float())

                if train:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                total_loss += loss.item() * imgs.size(0)
                all_logits.append(logits.sigmoid().float().cpu().detach())
                all_labels.append((original_lbls if train else lbls).cpu().detach())

        avg_loss = total_loss / len(loader.dataset)
        probs  = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()

        aucs = []
        per_class_auc = {}
        from sklearn.metrics import roc_auc_score
        for c in range(labels.shape[1]):
            col = labels[:, c]
            if col.sum() > 0 and col.sum() < len(col):
                auc = roc_auc_score(labels[:, c], probs[:, c])
                per_class_auc[ALL_CLASSES[c]] = round(auc, 3)
                aucs.append(auc)

        return avg_loss, np.mean(aucs) if aucs else 0.0, per_class_auc

    # Epoch Loop
    best_val = 0.0
    no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        # train_ds.transform = make_train_tf(256)

        tr_loss, tr_auc, _ = run_epoch(train_loader, train=True, scaler=scaler)
        val_loss, val_auc, per_class = run_epoch(val_loader, train=False, scaler=None)

        print("  Per-class AUCs:")
        for cls, auc in sorted(per_class.items(), key=lambda x: x[1]):
            print(f"    {cls:<20s} {auc:.3f}")

        scheduler.step()

        if val_auc > best_val:
            best_val = val_auc
            no_improve = 0
            torch.save(model.state_dict(), MODEL_OUTPUT_FILE)
            print("  ->  saved")
        else:
            no_improve += 1
            print(f"  (no improvement {no_improve}/{PATIENCE})")

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
              f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_auc={val_auc:.4f}")

        if no_improve >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print("Done. Best val AUC:", round(best_val, 4))
