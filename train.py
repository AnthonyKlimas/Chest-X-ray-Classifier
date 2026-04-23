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
import csv
import datetime
import math
import os
import glob
import sys
import time
import cv2
import numpy as np
import pandas as pd

import torch.nn as nn
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from skmultilearn.model_selection import IterativeStratification
from sklearn.preprocessing import MultiLabelBinarizer
from torch.amp import autocast
from tqdm import tqdm
from sklearn.metrics import roc_auc_score




from dataset import (
    make_value_tf, make_train_tf, worker_init_fn,
    print_dataset_parameters,   
    NIH_CXR8_CUSTOM_MEAN, NIH_CXR8_CUSTOM_STD,
    CLIP_LIMIT, TILE_GRID_SIZE, CXR8Dataset, CLAHETransform,
    PerImageStandardize, ALL_CLASSES
)

from swin_transformer_v2 import SwinTransformerV2
from visualization import GradCAM, visualize_class
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



### Tuning Parameters ###
# Training Control
NUM_EPOCHS = 48
PATIENCE = 14

# Loss Parameters
GAMMA_POS  = 1.0
GAMMA_NEG  = 2.0
ASYMMETRIC_CLIP = 0.05
# LABEL_SMOOTH = 0.0

ATTN_WARMUP_EPOCHS = 1

# BASE_LR = 5e-5
BASE_LR = 7e-5
# No pretraining for head
# Multiply the BASE_LR to compensate
HEAD_LR_MULTIPLIER = 5

LR_LAYER_DECAY = 0.8
WEIGHT_DECAY = 1e-2

# Sample extra from low appearing catagories
SAMPLER_POWER = 0.19


VIEW_POSITION_SCALE = 0.35

FEATURE_DROPOUT    = 0.2
CLASSIFIER_DROPOUT = 0.1

# Unfreeze the backbone stages slowly
UNFREEZE_SCHEDULE = {
     3: 0,
     7: 1,
    12: 2,
    18: 3,
}

# considering a schedule like this:
# UNFREEZE_SCHEDULE = {
#     2: 0,
#     5: 1,
#     9: 2,
#     14: 3,
# }


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

def get_class_attention(model, img_tensor, class_idx, device, view_id=0):
    model.eval()
    with torch.no_grad():
        feats = model.backbone.forward_features(img_tensor.unsqueeze(0).to(device))
        view_id = torch.tensor([view_id], dtype=torch.long, device=device)
        v = model.view_mlp(model.view_embed(view_id))
        gamma, beta = v.chunk(2, dim=-1)
        scale = torch.sigmoid(model.view_scale) * 2.0
        feats = feats * (1 + scale * gamma.unsqueeze(1)) + beta.unsqueeze(1)

        normed = model.attn_pool.norm(feats)
        attn = model.attn_pool.query(normed)                            # (1, N, num_classes)
        attn = torch.softmax(attn / model.attn_pool.temp.clamp(min=0.1), dim=1)
        attn_map = attn[0, :, class_idx]                                # (N,)

    H = W = int(attn_map.shape[0] ** 0.5)
    return attn_map.reshape(H, W).cpu().numpy()

# Logging and tuning purposes
def print_train_parameters():
    print("METADATA_CSV_PATH", METADATA_CSV_PATH)
    print("IMAGE_ROOT", IMAGE_ROOT)
    print("SSL_CKPT", SSL_CKPT)
    print("MODEL_OUTPUT_FILE", MODEL_OUTPUT_FILE)
    print("SAMPLER_POWER", SAMPLER_POWER)
    print("NUM_EPOCHS", NUM_EPOCHS)
    print("BASE_LR", BASE_LR)
    print("HEAD_LR_MULTIPLIER", HEAD_LR_MULTIPLIER)
    print("LR_LAYER_DECAY", LR_LAYER_DECAY)
    print("PATIENCE", PATIENCE)
    print("VIEW_POSITION_SCALE", VIEW_POSITION_SCALE)
    print("ASYMMETRIC_CLIP", ASYMMETRIC_CLIP)
    print("GAMMA_NEG", GAMMA_NEG)
    print("GAMMA_POS", GAMMA_POS)
    # print("LABEL_SMOOTH",LABEL_SMOOTH)
    print("ATTN_WARMUP_EPOCHS", ATTN_WARMUP_EPOCHS)
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
    df = df[["Image Index", "Finding Labels", "View Position", "Patient ID"]].copy()
    df = df[df["View Position"].isin(["PA", "AP"])].reset_index(drop=True)
    
    # No fillna — if something slipped past the filter above, we want to know
    df["view_id"] = df["View Position"].map({"PA": 0, "AP": 1})
    assert df["view_id"].isna().sum() == 0, "Unexpected view positions found after filter"
    df["view_id"] = df["view_id"].astype(int)
    
    df["labels"] = df["Finding Labels"].str.split("|")
    return df

def init_sampler(label_matrix, train_idx):
    active_cols = [i for i in range(len(ALL_CLASSES)) if i != NO_FINDING_COL]
    train_labels_active = label_matrix[train_idx][:, active_cols]
    class_weights = 1.0 / (train_labels_active.sum(axis=0) + 1e-6)
    sample_weights = (train_labels_active * class_weights).max(axis=1) ** SAMPLER_POWER
    sample_weights = np.clip(sample_weights, a_min=sample_weights[sample_weights > 0].min(), a_max=None)
    sample_weights = sample_weights / sample_weights.mean()
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

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


def init_split(df, label_matrix):
    patient_ids = df["Patient ID"].unique()

    patient_label_matrix = np.zeros((len(patient_ids), len(ALL_CLASSES)), dtype=int)
    patient_id_to_idx = {pid: i for i, pid in enumerate(patient_ids)}

    for img_idx, pid in enumerate(df["Patient ID"]):
        p = patient_id_to_idx[pid]
        patient_label_matrix[p] |= label_matrix[img_idx]

    # skmultilearn replacement — note it needs a sparse-compatible array
    stratifier = IterativeStratification(
        n_splits=2,                        # 2-fold gives one train/val split
        order=1,
        sample_distribution_per_fold=[0.15, 0.85],  # val=15%, train=85%
    )
    # Returns (larger_fold, smaller_fold) — val is first here
    val_patient_idx, train_patient_idx = next(
        stratifier.split(patient_ids.reshape(-1, 1), patient_label_matrix)
    )

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
    ckpt = ckpt["model"]

    # Fix label names
    ckpt = {
        k.replace("encoder.", "", 1): v
        for k, v in ckpt.items()
        if k.startswith("encoder.") and not k.startswith("encoder.mask_token")
    }
    ckpt = {
        k.replace("rpe_mlp", "cpb_mlp"): v
        for k, v in ckpt.items()
    }

    # Drop buffers that are recomputed at init from window_size
    ckpt = {
        k: v for k, v in ckpt.items()
        if "relative_coords_table" not in k
        and "relative_position_index" not in k
        and "attn_mask" not in k
    }

    missing, unexpected = model.backbone.load_state_dict(ckpt, strict=False)

    unexpected_missing = [k for k in missing if not any(tag in k for tag in EXPECTED_MISSING)]
    if len(unexpected_missing) > 0:
        print(f"WARNING: {len(unexpected_missing)} unexpected missing keys (expected only buffers):")
        for k in unexpected_missing:
            print(f"  {k}")

    # May be source of error if wrong ssl checkpoint file is used
    print("Loaded SSL checkpoint. Missing:", missing, "Unexpected:", unexpected)
    
    # no need to return ckpt, set in load_state_dict




def layer_unfreeze_epoch(layer_idx, schedule):
    if layer_idx < 0:
        return 1
    for epoch in sorted(schedule.keys()):
        if schedule[epoch] >= layer_idx:
            return epoch
    return 1

def make_lr_lambda(unfreeze_epoch, base_lr):
    eta_ratio = 0.0  # allow full decay to 0
    cosine_span = max(NUM_EPOCHS - unfreeze_epoch - UNFREEZE_WARMUP_EPOCHS, 1)

    def lr_lambda(epoch):
        e = epoch + 1
        if e < unfreeze_epoch:
            return 0.0
        warmup_frac = (e - unfreeze_epoch) / UNFREEZE_WARMUP_EPOCHS
        if warmup_frac < 1.0:
            return UNFREEZE_WARMUP_FACTOR + (1 - UNFREEZE_WARMUP_FACTOR) * warmup_frac
        cosine_e = e - unfreeze_epoch - UNFREEZE_WARMUP_EPOCHS
        cos = 0.5 * (1 + math.cos(math.pi * cosine_e / cosine_span))
        return eta_ratio + (1 - eta_ratio) * cos

    return lr_lambda



def verify_label_alignment(df, label_matrix, mlb, sample_indices=None):
    """
    Verifies that df and label_matrix are aligned after masking.
    Raises AssertionError immediately if anything is off.
    """
    assert len(df) == len(label_matrix), \
        f"Length mismatch: df={len(df)}, label_matrix={len(label_matrix)}"

    if sample_indices is None:
        n = len(df)
        sample_indices = sorted({0, n//4, n//2, 3*n//4, n-1})

    for i in sample_indices:
        raw = df.loc[i, "labels"]
        labels_list = raw.split("|") if isinstance(raw, str) else raw
        expected = mlb.transform([labels_list])[0]
        assert np.array_equal(label_matrix[i], expected), \
            f"Label mismatch at row {i}: got {label_matrix[i]}, expected {expected}"

    print(f"Label alignment verified across {len(sample_indices)} sampled rows.")


def verify_dataset_alignment(ds, label_matrix, idx_array):
    checkpoints = [0, len(ds)//2, len(ds)-1]
    for check_i in checkpoints:
        _, lbl, _ = ds[check_i]
        assert np.array_equal(lbl.numpy(), label_matrix[idx_array[check_i]]), \
            f"Dataset label mismatch at ds[{check_i}]"
    print(f"Dataset alignment verified at indices {checkpoints}.")


# Loss
class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_pos=1.0,
        gamma_neg=2.0,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        mask = torch.ones(NUM_CLASSES)
        mask[NO_FINDING_COL] = 0.0
        self.register_buffer("loss_mask", mask)

    def forward(self, logits, targets):
        # Sigmoid
        probs = torch.sigmoid(logits)

        # Positive / negative probabilities
        xs_pos = probs
        xs_neg = 1 - probs

        # Asymmetric clipping (ONLY negatives)
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Log terms
        log_pos = torch.log(xs_pos.clamp(min=self.eps))
        log_neg = torch.log(xs_neg.clamp(min=self.eps))

        # Basic CE loss
        loss = targets * log_pos + (1 - targets) * log_neg
        loss = loss * self.loss_mask


        # Asymmetric focusing
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    pt_pos = xs_pos * targets
                    pt_neg = xs_neg * (1 - targets)
                    pt = pt_pos + pt_neg
                    gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
                    focal_weight = (1 - pt) ** gamma
            else:
                pt_pos = xs_pos * targets
                pt_neg = xs_neg * (1 - targets)
                pt = pt_pos + pt_neg
                gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
                focal_weight = (1 - pt) ** gamma

            loss *= focal_weight

        active = self.loss_mask.sum() * logits.shape[0]
        return -loss.sum() / active    

class ClassSpecificAttnPool(nn.Module):
    """
    Each class learns its own spatial attention query.
    Input:  feats  (B, N, C)
    Output: pooled (B, num_classes, C)
    """
    def __init__(self, C, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(C)
        # One query per class
        self.query = nn.Linear(C, num_classes, bias=False)
        self.temp   = nn.Parameter(torch.ones(1))

    def forward(self, feats):
        # feats: (B, N, C)
        feats = self.norm(feats)
        # Attention logits per class: (B, N, num_classes)
        attn = self.query(feats) / self.temp.clamp(min=0.1)
        attn = torch.softmax(attn, dim=1)           # softmax over N
        # Weighted sum per class: (B, num_classes, C)
        pooled = torch.einsum("bnc,bnk->bkc", feats, attn)
        return pooled  # (B, num_classes, C)


class SwinWithView(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        C = backbone.norm.normalized_shape[0]
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.num_classes = num_classes
        self.use_attention = True 

        # --- Class-specific pooling replaces attn_pool ---
        self.attn_pool = ClassSpecificAttnPool(C, num_classes)

        # View conditioning (unchanged)
        self.view_embed = nn.Embedding(2, 32)
        self.view_mlp   = nn.Sequential(
            nn.Linear(32, 128),
            nn.GELU(),
            nn.Linear(128, C * 2)
        )
        self.view_scale = nn.Parameter(torch.tensor(VIEW_POSITION_SCALE))

        nn.init.zeros_(self.view_mlp[-1].weight)
        nn.init.zeros_(self.view_mlp[-1].bias)

        # Per-class head: each class gets its own (C -> 1) projection
        # implemented efficiently as a single Linear(C, num_classes)
        self.head = nn.Linear(C, num_classes)

        # self.head = nn.Sequential(
        #     nn.LayerNorm(C),
        #     nn.Dropout(FEATURE_DROPOUT),
        #     nn.Linear(C, 512),
        #     nn.GELU(),
        #     nn.Dropout(CLASSIFIER_DROPOUT),
        #     nn.Linear(512, 1),   # applied per class independently
        # )

    def forward(self, x, view_id):
        feats = self.backbone.forward_features(x)   # (B, N, C)

        # View conditioning: modulate the spatial tokens before pooling
        # so each class-query sees view-conditioned features
        v = self.view_mlp(self.view_embed(view_id))  # (B, C*2)
        gamma, beta = v.chunk(2, dim=-1)             # each (B, C)
        scale = torch.sigmoid(self.view_scale) * 2.0
        # Broadcast over N
        feats = feats * (1 + scale * gamma.unsqueeze(1)) + beta.unsqueeze(1)

        if self.use_attention:
            pooled = self.attn_pool(feats)  # (B, K, C)
        else:
            B, N, C = feats.shape
            pooled = feats.mean(dim=1, keepdim=True)        # (B, 1, C)
            pooled = pooled.expand(-1, self.num_classes, -1)  # (B, K, C)

        # Apply head to each class's feature vector
        # Reshape to (B * num_classes, C), run head, reshape back
        B, K, C = pooled.shape
        pooled_flat = pooled.reshape(B * K, C)
        logits = self.head(pooled_flat).squeeze(-1)  # (B * num_classes)
        logits = logits.reshape(B, K)                # (B, num_classes)

        return logits

# Param Groups
def init_param_groups(model, base_lr=1e-4, decay=0.8, schedule=None):
    schedule = schedule or {}
    groups = []
    seen = set()

    def add(params, lr, layer_idx, weight_decay=1e-2):
        wd, no_wd = [], []
        for p in params:
            pid = id(p)
            if pid in seen: continue
            seen.add(pid)
            (wd if p.ndim > 1 else no_wd).append(p)

        ue = layer_unfreeze_epoch(layer_idx, schedule)
        for bucket, wdv in [(wd, weight_decay), (no_wd, 0.0)]:
            if bucket:
                groups.append({
                    "params": bucket,
                    "lr": lr,
                    "base_lr": lr, # stored for lambda scaling
                    "layer_idx": layer_idx,
                    "unfreeze_epoch": ue,
                    "weight_decay": wdv,
                })

    layers = list(model.backbone.layers)
    n_layers = len(model.backbone.layers)
    for i, layer in enumerate(model.backbone.layers):
        add(layer.parameters(), base_lr * (decay ** (n_layers - 1 - i)), i)

    add(model.head.parameters(),       base_lr * HEAD_LR_MULTIPLIER, -1)
    add(model.view_embed.parameters(), base_lr, -1)
    add(model.view_mlp.parameters(),   base_lr, -1)
    add(model.attn_pool.parameters(), base_lr * 2, -1)
    add([model.view_scale], base_lr, -1)

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

    verify_label_alignment(df, label_matrix, mlb)



    # Split
    # split by Patient so value tests hasn't been trained on the same patients
    # This causes an auc decrease of about 0.01 but is a more accurate test
    # Aggregate labels to patient level for stratified splitting
    train_idx, value_idx = init_split(df, label_matrix=label_matrix)

    # Transforms
    value_tf   = make_value_tf(256)
    train_tf = make_train_tf(256)
    
    train_ds = CXR8Dataset(df, label_matrix, train_idx, train_tf, path_lookup)
    value_ds = CXR8Dataset(df, label_matrix, value_idx,   value_tf,   path_lookup)
        
    # After building train_ds, verify alignment:
    verify_dataset_alignment(train_ds, label_matrix, train_idx)


    # Sampler
    # Exclude No Finding from sampler weighting — it's masked in loss
    active_cols = [i for i in range(len(ALL_CLASSES)) if i != NO_FINDING_COL]
    train_labels_active = label_matrix[train_idx][:, active_cols]

    class_counts  = train_labels_active.sum(axis=0)
    class_weights = 1.0 / (class_counts + 1e-6)

    sample_weights = (train_labels_active * class_weights).max(axis=1)
    sample_weights = sample_weights ** SAMPLER_POWER

    # "No Finding"-only images now have weight 0 here, so the clip
    # brings them up to the minimum positive weight — they're still seen,
    # but not over-represented relative to true disease cases
    sample_weights = np.clip(
        sample_weights,
        a_min=sample_weights[sample_weights > 0].min(),
        a_max=None,
    )
    sample_weights = sample_weights / sample_weights.mean()
    
    sampler = init_sampler(label_matrix=label_matrix, train_idx=train_idx)
    

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
                
    param_group = init_param_groups(model, base_lr=BASE_LR,
                                    decay=LR_LAYER_DECAY, schedule=UNFREEZE_SCHEDULE)
    optimizer = torch.optim.AdamW(param_group, weight_decay=WEIGHT_DECAY)


    lambdas = [make_lr_lambda(g["unfreeze_epoch"], g["base_lr"])
            for g in optimizer.param_groups]
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    # Freeze backbone layers whose unfreeze_epoch > 1
    for layer, idx in layer_to_idx.items():
        if layer_unfreeze_epoch(idx, UNFREEZE_SCHEDULE) > 1:
            for p in layer.parameters():
                p.requires_grad = False
   

    # Better logging
    log_path = "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "tr_loss", "tr_auc", "val_loss", "val_auc"] + ALL_CLASSES)

    

    # scaler = GradScaler(device="cuda")
    criterion = AsymmetricLoss(
        gamma_pos=GAMMA_POS,
        gamma_neg=GAMMA_NEG,
        clip=ASYMMETRIC_CLIP,
        # label_smooth=LABEL_SMOOTH,
    )

    ###
    ### Training cycle ###
    ###

    def run_epoch(loader, train=True):

        model.train() if train else model.eval()
        total_loss = 0.0
        n_samples = 0
        all_logits, all_labels = [], []


        with torch.set_grad_enabled(train):
            for imgs, lbls, views in tqdm(loader, desc="train" if train else "val ", leave=False, mininterval=10.0):
                imgs  = imgs.to(device, non_blocking=True)
                lbls  = lbls.to(device, non_blocking=True)
                views = views.to(device, non_blocking=True)
                n_samples += imgs.size(0)

                if train:
                    optimizer.zero_grad()
                
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                # with autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    logits = model(imgs, views)
                    loss = criterion(logits, lbls)

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                all_logits.append(logits.sigmoid().float().cpu().detach())
                all_labels.append(lbls.detach().cpu())



        avg_loss = total_loss / n_samples
        probs  = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()

        per_class_auc = {}
        aucs = []
        for c in range(labels.shape[1]):
            if c == NO_FINDING_COL:
                continue
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



   
    best_val = 0.0
    no_improve = 0
    
    ### Training loop ###
    for epoch in range(1, NUM_EPOCHS + 1):
        # Unfreeze layers at scheduled epoch
        for layer, idx in layer_to_idx.items():
            if epoch == layer_unfreeze_epoch(idx, UNFREEZE_SCHEDULE):
                for p in layer.parameters():
                    p.requires_grad = True


        # Disable attention early
        if epoch <= ATTN_WARMUP_EPOCHS:
            model.use_attention = False
        else:
            model.use_attention = True

        ### Train epoch ###
        tr_loss, tr_auc, _ = run_epoch(train_loader, train=True)
        
        scheduler.step()
        torch.cuda.empty_cache()
        time.sleep(HARDWARE_PITY)
        
        ### Value epoch ###
        val_loss, val_auc, per_class = run_epoch(val_loader, train=False)

        print("  Per-class AUCs:")
        for cls, auc in sorted(per_class.items(), key=lambda x: x[1]):
            print(f"    {cls:<20s} {auc:.3f}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, tr_loss, tr_auc, val_loss, val_auc] +
                            [per_class.get(c, "") for c in ALL_CLASSES])

        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val": best_val,
                "no_improve": no_improve,
            }, f"checkpoint_epoch{epoch:02d}.pth")
        if val_auc > best_val:
            best_val = val_auc
            no_improve = 0
            torch.save(model.state_dict(), MODEL_OUTPUT_FILE)
            print("  ->  saved")
        else:
            no_improve += 1
            print(f"  (no improvement {no_improve}/{PATIENCE})")

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
            f"train_loss={tr_loss:.4f}  train_auc={tr_auc:.4f}  "
            f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f}")

        head_lr  = next(g["lr"] for g in optimizer.param_groups if g.get("layer_idx") == -1)
        layer3_lr = next(g["lr"] for g in optimizer.param_groups if g.get("layer_idx") == 3)
        print(f"  head_lr={head_lr:.2e}  layer3_lr={layer3_lr:.2e}")

        # for g in optimizer.param_groups:
        #     print(g["layer_idx"], g["lr"])

        if no_improve >= PATIENCE and epoch > max(UNFREEZE_SCHEDULE.keys()) + UNFREEZE_WARMUP_EPOCHS:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print("Done. Best val AUC:", round(best_val, 4))
    
    # Inspect learned view conditioning
    model.load_state_dict(torch.load(MODEL_OUTPUT_FILE, map_location=device))
    print("view_scale:", torch.sigmoid(model.view_scale).item() * 2.0)
    
    # Print the attn_temp to see if attention pooling sharpened
    print("attn_temp:", model.attn_pool.temp.item())

    model.eval()

    gradcam = GradCAM(model, device)

    # Grab one validation image
    for vis_i in range(len(value_ds)):
        img_tensor, lbl, view = value_ds[vis_i]
        if lbl.sum() > 0 and lbl[NO_FINDING_COL] == 0:
            img_np = cv2.imread(path_lookup[df.iloc[value_idx[vis_i]]["Image Index"]], cv2.IMREAD_GRAYSCALE)
            break
    else:
        print("Warning: no suitable validation image found for visualization")
        gradcam.remove()
        sys.exit()


    for class_idx in range(NUM_CLASSES):
        if lbl[class_idx] == 1:   # only visualize positive classes for this image
            visualize_class(model, img_tensor, img_np, class_idx,
                            device, gradcam, view_id=int(view),
                            save_path=f"viz_{ALL_CLASSES[class_idx]}.png")

    gradcam.remove()
