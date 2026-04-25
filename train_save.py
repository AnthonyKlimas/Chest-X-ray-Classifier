"""
Author: Nicholas J. Calabro
Group 5
Coyyright 2026


Current Model:

Pretrained with SimMIM, Microsoft's SwinV2 is used as a backbone.
The backbone has an unfreeze schedule to give it time to adjust to
a new dataset.

A MLP is used as the head. 



Note one definite inaccuracy: Hernia has such few entries
that the value test will not yield a valid result. 
"""
import datetime
import os
import glob
import time
import csv
import numpy as np
import pandas as pd


from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedShuffleSplit



from dataset import (
    make_value_tf, make_train_tf, worker_init_fn,
    print_dataset_parameters,   
    NIH_CXR8_CUSTOM_MEAN, NIH_CXR8_CUSTOM_STD,
    CLIP_LIMIT, TILE_GRID_SIZE, CXR8Dataset, CLAHETransform,
    PerImageStandardize, ALL_CLASSES
)

from swin_transformer_v2 import SwinTransformerV2

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


### Paths (must resolve to train) ###

# Metadata file
METADATA_CSV_PATH = "../chest_xray_dataset/CXR8/Data_Entry_2017_v2020.csv"

# Expects unziped subdirectories containing original png image filenames
IMAGE_ROOT = "../chest_xray_dataset/CXR8/images_preprocessed"
# IMAGE_ROOT = "../chest_xray_dataset/CXR8/image"

# Self Supervized Learning Checkpoint filepath
SSL_CKPT = "../chest_xray_dataset/swinv2_small_1k_500k.pth"

MODEL_OUTPUT_FILE = "swin_cxr8_best.pth"

LOG_PATH = "training_log.csv"

### Tuning Parameters ###
# Training Control
NUM_EPOCHS = 48
PATIENCE = 14

# Loss Parameters
GAMMA_POS  = 1.0
GAMMA_NEG  = 3.0
ASYMMETRIC_CLIP = 0.03

# BASE_LR = 5e-5
BASE_LR = 7e-5
# No pretraining for head
# Multiply the BASE_LR to compensate
HEAD_LR_MULTIPLIER = 6

LR_LAYER_DECAY = 0.8
WEIGHT_DECAY = 1e-2

# Sample extra from low appearing catagories
SAMPLER_POWER = 0.17

# Warmup backbone; previously trained
WARMUP_EPOCHS = 3
WARMUP_START_FACTOR = 0.3
WARMUP_END_FACTOR = 1.0

EMA_DECAY = 0.9988
CHECKPOINT_INTERVAL = 5

VIEW_POSITION_SCALE = 0.35

FEATURE_DROPOUT    = 0.2
CLASSIFIER_DROPOUT = 0.1

# Unfreeze the backbone stages slowly
UNFREEZE_SCHEDULE = {
     6: 3,
    11: 2,
    17: 1,
    24: 0,
}
UNFREEZE_WARMUP_EPOCHS = 5
UNFREEZE_WARMUP_FACTOR = 0.1

# seconds to sleep after training
# set to 0 if not concerned about hardware overheating
HARDWARE_PITY = 45


### Data Loader Parameters ###
# Ran without locking when workers were 2 and 2
# CPU will bottleneck less with higher workers, may deadlock (immediately)
LOADER_WORKERS_TRAIN = 10
LOADER_WORKERS_VALUE   = 2
BATCH_SIZE_VAL = 16
BATCH_SIZE_TRAIN = 16

PREFETECH_FACTOR = 2
PERSISTENT_WORKERS = True

# Minimum number of value needed to evaluate a catagory
MIN_VAL_POSITIVES = 50


### Calculated Parameters ###
NO_FINDING_COL = ALL_CLASSES.index("No Finding")
NUM_CLASSES = len(ALL_CLASSES)

# Check point file labels that aren't required
EXPECTED_MISSING = {"relative_coords_table", "relative_position_index", "attn_mask"}


# Logging and tuning purposes
def print_train_parameters():
    print("METADATA_CSV_PATH", METADATA_CSV_PATH)
    print("IMAGE_ROOT", IMAGE_ROOT)
    print("SSL_CKPT", SSL_CKPT)
    print("MODEL_OUTPUT_FILE", MODEL_OUTPUT_FILE)
    print("SAMPLER_POWER", SAMPLER_POWER)
    print("WARMUP_EPOCHS", WARMUP_EPOCHS)
    print("WARMUP_START_FACTOR", WARMUP_START_FACTOR)
    print("WARMUP_END_FACTOR", WARMUP_END_FACTOR)
    print("NUM_EPOCHS", NUM_EPOCHS)
    print("BASE_LR", BASE_LR)
    print("HEAD_LR_MULTIPLIER", HEAD_LR_MULTIPLIER)
    print("LR_LAYER_DECAY", LR_LAYER_DECAY)
    print("PATIENCE", PATIENCE)
    print("VIEW_POSITION_SCALE", VIEW_POSITION_SCALE)
    print("ASYMMETRIC_CLIP", ASYMMETRIC_CLIP)
    print("GAMMA_NEG", GAMMA_NEG)
    print("GAMMA_POS", GAMMA_POS)
    print("UNFREEZE_WARMUP_EPOCHS", UNFREEZE_WARMUP_EPOCHS)
    print("UNFREEZE_WARMUP_FACTOR",UNFREEZE_WARMUP_FACTOR)
    print("WEIGHT_DECAY", WEIGHT_DECAY)
    print("FEATURE_DROPOUT", FEATURE_DROPOUT)
    print("CLASSIFIER_DROPOUT", CLASSIFIER_DROPOUT)
    print("BATCH_SIZE_VAL", BATCH_SIZE_VAL)
    print("BATCH_SIZE_TRAIN", BATCH_SIZE_TRAIN)
    print("HARDWARE_PITY", HARDWARE_PITY)
    print("UNFREEZE_SCHEDULE", UNFREEZE_SCHEDULE )
    print("TRAIN_LOADER_WORKERS", LOADER_WORKERS_TRAIN )
    print("VALUE_LOADER_WORKERS", LOADER_WORKERS_VALUE)
    print("PREFETECH_FACTOR", PREFETECH_FACTOR)
    print("PERSISTENT_WORKERS", PERSISTENT_WORKERS)
    print(datetime.datetime.now())


def init_metadata(path):
    df = pd.read_csv(path)
    df = df[["Image Index", "Finding Labels",
             "View Position", "Patient ID"]].copy()
    df["view_id"] = df["View Position"].map({"PA": 0, "AP": 1}).fillna(0).astype(int)
    df = df[df["View Position"].isin(["PA", "AP"])].reset_index(drop=True)
    df["labels"] = df["Finding Labels"].str.split("|")
    
    return df

def init_device():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}]", torch.cuda.get_device_name(i))
    print("Using device:", device)
    print("**Device: ", device)
    print(torch.cuda.get_device_capability())
    torch.backends.cudnn.benchmark = True

    return device


def layer_unfreeze_epoch(layer_idx, schedule):
    if layer_idx < 0:        # head, view embed, attn_pool — always live
        return 1
    for epoch in sorted(schedule.keys()):
        if layer_idx >= schedule[epoch]:
            return epoch
    return 1

class UnfreezeScheduler:
    def __init__(self, layer_to_idx, optimizer, schedule, warmup_epochs):
        self.epoch = 1
        self.optimizer = optimizer
        self.schedule = schedule
        self.layer_to_idx = layer_to_idx
        self.warmup_epochs = warmup_epochs
        # freeze layers
        for layer, idx in layer_to_idx.items():
            for p in layer.parameters():
                p.requires_grad = False

    # Unfreeze layers per schedule
    def step(self, group_warmup_remaining):
        newly_unfrozen = set()
        if self.epoch in self.schedule:
            if group_warmup_remaining:
                print(f"WARNING: Unfreezing at epoch {self.epoch} but warmup still active for layers: {list(group_warmup_remaining.keys())}")
            
            threshold = self.schedule[self.epoch]
            # Check no warmup is still in progress
            for layer, idx in self.layer_to_idx.items():
                if idx >= threshold:
                    for p in layer.parameters():
                        if not p.requires_grad:
                            p.requires_grad = True
                            newly_unfrozen.add(idx)
            for group in self.optimizer.param_groups:
                lidx = group.get("layer_idx", -1)
                if lidx in newly_unfrozen:
                    # Use cosine-decayed peer LR, not the original base_lr
                    ref = next(g for g in self.optimizer.param_groups
                            if g.get("layer_idx") == -1)
                    cosine_scale = ref["lr"] / ref["base_lr"]
                    group["lr"] = group["base_lr"] * cosine_scale
                    group_warmup_remaining[lidx] = self.warmup_epochs
        self.epoch += 1

def init_split(df, label_matrix):
    patient_ids = df["Patient ID"].unique()

    patient_label_matrix = np.zeros((len(patient_ids), len(ALL_CLASSES)), dtype=int)
    patient_id_to_idx = {pid: i for i, pid in enumerate(patient_ids)}
    for img_idx, row in df.iterrows():
        p = patient_id_to_idx[row["Patient ID"]]
        patient_label_matrix[p] |= label_matrix[img_idx]

    # Collapse multilabel to a single stratification key via label combination hash
    # Rare combos get lumped into a single "other" bin to avoid singleton strata
    combo_strings = ["_".join(map(str, row)) for row in patient_label_matrix]
    from collections import Counter
    counts = Counter(combo_strings)
    MIN_COMBO_COUNT = 2
    strat_labels = [c if counts[c] >= MIN_COMBO_COUNT else "__other__" for c in combo_strings]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_patient_idx, val_patient_idx = next(sss.split(patient_ids, strat_labels))

    train_patients = set(patient_ids[train_patient_idx])
    value_patients = set(patient_ids[val_patient_idx])

    train_idx = df[df["Patient ID"].isin(train_patients)].index.to_numpy()
    value_idx = df[df["Patient ID"].isin(value_patients)].index.to_numpy()

    for split_name, idx in [("train", train_idx), ("val", value_idx)]:
        n_hernia = label_matrix[idx, ALL_CLASSES.index("Hernia")].sum()
        print(f"{split_name} Hernia positives: {n_hernia}")

    return train_idx, value_idx

# Loads the SSL checkpoint from the path SSL_CKPT
def init_ckpt(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    # Already flat — no wrapper key, no encoder prefix to strip
    # Just drop the recomputed buffers and the classification head
    ckpt = {
        k.replace("rpe_mlp", "cpb_mlp"): v
        for k, v in ckpt.items()
        if "relative_coords_table" not in k
        and "relative_position_index" not in k
        and "attn_mask" not in k
        and k not in ("head.weight", "head.bias")  # your head is different
    }

    missing, unexpected = model.backbone.load_state_dict(ckpt, strict=False)

    unexpected_missing = [k for k in missing if not any(tag in k for tag in EXPECTED_MISSING)]
    if unexpected_missing:
        print(f"WARNING: {len(unexpected_missing)} unexpected missing keys:")
        for k in unexpected_missing:
            print(f"  {k}")

# Loss
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=1, gamma_neg=4, clip=0.05,
                 eps=1e-8,  label_smooth=0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.label_smooth = label_smooth


    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        # Clip negative probabilities
        if self.clip > 0:
            probs_neg = (1 - probs - self.clip).clamp(min=0)
        else:
            probs_neg = 1 - probs

        # Asymmetric focusing
        pos_focal = (1 - probs) ** self.gamma_pos
        neg_focal = probs ** self.gamma_neg

        if self.label_smooth > 0:
                    targets = targets * (1 - self.label_smooth) + 0.5 * self.label_smooth

        # Loss
        loss_pos = targets * torch.log(probs.clamp(min=self.eps)) * pos_focal
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps)) * neg_focal

        loss = -(loss_pos + loss_neg).mean()
        return loss


# Model Wrapper
class SwinWithView(torch.nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        C = backbone.norm.normalized_shape[0]
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.attn_pool = torch.nn.Sequential(
            torch.nn.LayerNorm(C),
            torch.nn.Linear(C, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 1)
        )
        
        self.view_embed = torch.nn.Embedding(2, 32)
        self.view_mlp = torch.nn.Sequential(
            torch.nn.Linear(32, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, C * 2)
        )
        self.attn_temp  = torch.nn.Parameter(torch.tensor(1.0))
        self.view_scale = torch.nn.Parameter(torch.tensor(VIEW_POSITION_SCALE))
        nn.init.zeros_(self.view_mlp[-1].weight)
        nn.init.zeros_(self.view_mlp[-1].bias)
        nn.init.zeros_(self.attn_pool[-1].weight)
        nn.init.zeros_(self.attn_pool[-1].bias)

        # Multi-classifier head
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(C),
            torch.nn.Dropout(FEATURE_DROPOUT),
            torch.nn.Linear(C, 768),
            torch.nn.GELU(),
            torch.nn.Dropout(CLASSIFIER_DROPOUT),
            torch.nn.Linear(768, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, num_classes),
        )

    def forward(self, x, view_id):
        feats = self.backbone.forward_features(x)  # (B, N, C)

        if feats.ndim == 3:
            B, N, C = feats.shape
            attn = self.attn_pool(feats).squeeze(-1)
            temp = self.attn_temp.clamp(min=0.1)
            attn = torch.softmax(attn / temp, dim=1)
            feats = (feats * attn.unsqueeze(-1)).sum(dim=1)


        v = self.view_mlp(self.view_embed(view_id))
        gamma, beta = v.chunk(2, dim=-1)

        scale = torch.sigmoid(self.view_scale) * 2.0
        feats = feats * (1 + scale * gamma) + beta
        return self.head(feats)


# Param Groups
def init_param_groups(model, base_lr=1e-4, decay=0.8):
    groups = []
    seen = set()

    def add(params, lr, layer_idx, weight_decay=1e-2):
        wd, no_wd = [], []

        for p in params:
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)

            if p.ndim <= 1:
                no_wd.append(p)
            else:
                wd.append(p)

        if wd:
            groups.append({
                "params": wd,
                "lr": lr,
                "layer_idx": layer_idx,
                "weight_decay": weight_decay
            })

        if no_wd:
            groups.append({
                "params": no_wd,
                "lr": lr,
                "layer_idx": layer_idx,
                "weight_decay": 0.0
            })

    layers = list(model.backbone.layers)

    for i, layer in enumerate(reversed(layers)):
        lr = base_lr * (decay ** i)
        layer_idx = len(layers) - 1 - i
        add(layer.parameters(), lr, layer_idx)

    add(model.head.parameters(), base_lr * HEAD_LR_MULTIPLIER, layer_idx=-1)
    add(model.view_embed.parameters(), base_lr, -1)
    add(model.view_mlp.parameters(), base_lr, -1)
    add(model.attn_pool.parameters(), base_lr, -1)
    add([model.attn_temp, model.view_scale], base_lr, -1)

    leftovers = [p for p in model.parameters() if id(p) not in seen]
    if leftovers:
        add(leftovers, base_lr * (decay ** len(layers)), -2)

    return groups


# Main Model Driver
if __name__ == "__main__":

    # For logging & tuning purposes
    print_train_parameters()
    print_dataset_parameters()

    # GPU prep and init
    device = init_device()

    # Load metadata
    df = init_metadata(METADATA_CSV_PATH)
    

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
    # split by Patient so value tests hasn't been trained on the same patients
    # This causes an auc decrease of about 0.01 but is a more accurate test
    # Aggregate labels to patient level for stratified splitting
    train_idx, value_idx = init_split(df, label_matrix=label_matrix)

    # Transforms
    value_tf   = make_value_tf(256)
    train_tf = make_train_tf(256)
    
    train_ds = CXR8Dataset(df, label_matrix, train_idx, train_tf, path_lookup)
    value_ds   = CXR8Dataset(df, label_matrix, value_idx,   value_tf,   path_lookup)
    
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
        num_workers=LOADER_WORKERS_TRAIN,
        worker_init_fn=worker_init_fn,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETECH_FACTOR,
    )

    val_loader = DataLoader(
        value_ds,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        num_workers=LOADER_WORKERS_VALUE,
        worker_init_fn=worker_init_fn,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETECH_FACTOR,
    )

    # Model
    base = SwinTransformerV2(
        img_size=256,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        # Param refers to gradient checkpoint, not SSL checkpoint
        use_checkpoint=False,
    )

    model = SwinWithView(backbone=base, num_classes=NUM_CLASSES).to(device)

    with torch.no_grad():
        x = torch.randn(1, 3, 256, 256).to(device)
        feats = model.backbone.forward_features(x)
        print("Backbone output shape:", feats.shape)


    layer_to_idx = {
        layer: i
        for i, layer in enumerate(model.backbone.layers)
    }

    # Model Training Checkpoint
    init_ckpt(model=model, path=SSL_CKPT)

    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay=EMA_DECAY))

                
    param_group = init_param_groups(model, base_lr=BASE_LR, decay=LR_LAYER_DECAY)

    optimizer = torch.optim.AdamW(param_group, weight_decay=WEIGHT_DECAY)    

   
    # scaler = GradScaler(device="cuda")
    criterion = AsymmetricLoss(
        gamma_pos=GAMMA_POS,
        gamma_neg=GAMMA_NEG,
        clip=ASYMMETRIC_CLIP,
        label_smooth=0.05,
    )

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

    # Unfreeze SwinV2 stages to warmup backbone
    unfreeze_scheduler = UnfreezeScheduler(
        layer_to_idx=layer_to_idx,
        optimizer=optimizer,
        schedule=UNFREEZE_SCHEDULE,
        warmup_epochs=UNFREEZE_WARMUP_EPOCHS
    )

    # scheduler = torch.optim.lr_scheduler.SequentialLR(
    #     optimizer,
    #     schedulers=[warmup_scheduler, cosine_scheduler],
    #     milestones=[WARMUP_EPOCHS]
    # )


    # Training Loop
    def run_epoch(loader, train=True, eval_model=None):
        active_model = eval_model if (not train and eval_model is not None) else model
        active_model.train() if train else active_model.eval()
        # model.train() if train else model.eval()
        total_loss = 0.0
        n_samples = 0
        all_logits, all_labels = [], []

        with torch.set_grad_enabled(train):
            for imgs, lbls, views in tqdm(loader, desc="train" if train else "val ", leave=False, mininterval=10.0):
                imgs  = imgs.to(device, non_blocking=True)
                lbls  = lbls.to(device, non_blocking=True)
                views = views.to(device, non_blocking=True)
                n_samples += imgs.size(0)

                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                # with autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    logits = active_model(imgs, views)
                    loss = criterion(logits, lbls)

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    ema_model.update_parameters(model)

                total_loss += loss.item() * imgs.size(0)
                all_logits.append(logits.sigmoid().float().cpu().detach())
                all_labels.append(lbls.detach().cpu())



        avg_loss = total_loss / n_samples
        probs  = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()

        per_class_auc = {}
        aucs = []
        for c in range(labels.shape[1]):
            col = labels[:, c]
            n_pos = col.sum()
            if n_pos >= MIN_VAL_POSITIVES and n_pos < len(col):
                auc = roc_auc_score(col, probs[:, c])
                per_class_auc[ALL_CLASSES[c]] = round(auc, 3)
                aucs.append(auc)
            elif n_pos > 0:
                # Still log it, just don't include in mean
                per_class_auc[ALL_CLASSES[c]] = round(roc_auc_score(col, probs[:, c]), 3)

        return avg_loss, np.mean(aucs) if aucs else 0.0, per_class_auc



    # After building optimizer, store intended LRs once
    for group in optimizer.param_groups:
        group["base_lr"] = group["lr"]

    # Epoch Loop
    # Initialize CSV log
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "tr_loss", "tr_auc", "val_loss", "val_auc"] + ALL_CLASSES)

    best_val = 0.0
    no_improve = 0
    group_warmup_remaining = {}

    for epoch in range(1, NUM_EPOCHS + 1):

        tr_loss, tr_auc, _ = run_epoch(train_loader, train=True)
        torch.cuda.empty_cache()
        time.sleep(HARDWARE_PITY)

        # Evaluate with EMA model
        val_loss, val_auc, per_class = run_epoch(val_loader, train=False,
                                                eval_model=ema_model.module)

        print("  Per-class AUCs:")
        for cls, auc in sorted(per_class.items(), key=lambda x: x[1]):
            print(f"    {cls:<20s} {auc:.3f}")

        # Write CSV row
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, tr_loss, tr_auc, val_loss, val_auc] +
                            [per_class.get(c, "") for c in ALL_CLASSES])

        # Periodic full checkpoint
        if epoch % CHECKPOINT_INTERVAL == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_model": ema_model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val,
                "no_improve": no_improve,
            }, f"checkpoint_epoch{epoch:02d}.pth")

        if val_auc > best_val:
            best_val = val_auc
            no_improve = 0
            torch.save(ema_model.module.state_dict(), MODEL_OUTPUT_FILE)
            print("  ->  saved")
        else:
            no_improve += 1
            print(f"  (no improvement {no_improve}/{PATIENCE})")

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
            f"train_loss={tr_loss:.4f}  train_auc={tr_auc:.4f}  "
            f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f}")

        print(f"  head_lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"layer3_lr={next(g['lr'] for g in optimizer.param_groups if g.get('layer_idx')==3):.2e}")

        unfreeze_scheduler.step(group_warmup_remaining)
        if epoch <= WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        for group in optimizer.param_groups:
            lidx = group.get("layer_idx", -1)
            if lidx in group_warmup_remaining:
                epochs_done = UNFREEZE_WARMUP_EPOCHS - group_warmup_remaining[lidx]
                scale = UNFREEZE_WARMUP_FACTOR + (1 - UNFREEZE_WARMUP_FACTOR) * (epochs_done / UNFREEZE_WARMUP_EPOCHS)
                group["lr"] = group["base_lr"] * scale

        for k in list(group_warmup_remaining):
            group_warmup_remaining[k] -= 1
            if group_warmup_remaining[k] <= 0:
                del group_warmup_remaining[k]

        # Guard: don't stop before all unfreeze events have had time to stabilize
        if no_improve >= PATIENCE and epoch > max(UNFREEZE_SCHEDULE.keys()) + WARMUP_EPOCHS:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print("Done. Best val AUC:", round(best_val, 4))
    
    # Inspect learned view conditioning
    model.load_state_dict(torch.load(MODEL_OUTPUT_FILE, map_location=device))
    print("view_scale:", torch.sigmoid(model.view_scale).item() * 2.0)
    
    # Print the attn_temp to see if attention pooling sharpened
    print("attn_temp:", model.attn_temp.item())
