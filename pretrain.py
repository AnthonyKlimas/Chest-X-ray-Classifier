"""
SimMIM Self-Supervised Pretraining for SwinV2 on Chest X-rays

Fixes vs original simmim.py:
  1. Learned mask token instead of zero-fill
  2. Per-patch normalized reconstruction targets
  3. FPN decoder with skip connections from all 4 backbone stages
  4. Checkpoint format compatible with train.py init_ckpt
  5. Full resumable checkpoints (model + optimizer + scaler + epoch)
  6. use_checkpoint=False (48 GB VRAM — no need for activation checkpointing)

CXR-specific improvements:
  - CLAHE + per-image standardization matches supervised train.py preprocessing
  - No vertical flip, no color jitter (anatomically wrong / grayscale)
  - Mild random rotation + horizontal flip only

Multi-GPU:
  torchrun --nproc_per_node=8 pretrain.py
Single-GPU:
  python pretrain.py
"""

import datetime
import glob
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from dataset import CLIP_LIMIT, TILE_GRID_SIZE, CLAHETransform, PerImageStandardize
from swin_transformer_v2 import SwinTransformerV2


# Paths
IMAGE_ROOT  = "../chest_xray_dataset/CXR8/images_preprocessed"
OUTPUT_CKPT = "../chest_xray_dataset/simmim_swinv2_cxr_backbone.pth"
RESUME_CKPT = None   # set to a full checkpoint path to resume training

# Architecture (must match train.py)
IMG_SIZE    = 384
PATCH_SIZE  = 4
IN_CHANS    = 3
EMBED_DIM   = 96
DEPTHS      = [2, 2, 18, 2]
NUM_HEADS   = [3, 6, 12, 24]
WINDOW_SIZE = 8

# Pretraining
MASK_RATIO   = 0.60
DECODER_DIM  = 256   # internal channel width of FPN decoder

# Per GPU.  8 GPUs × 32 = effective batch 256.
BATCH_SIZE   = 32
ACCUM_STEPS  = 8     # increase to further multiply effective batch

# Square-root scaling from reference (1e-4 @ bs=32 → 2e-4 @ bs=256)
BASE_LR        = 2e-4
WEIGHT_DECAY   = 0.05
WARMUP_EPOCHS  = 10
NUM_EPOCHS     = 100
NUM_WORKERS    = 10
PRINT_FREQ     = 100
SAVE_EVERY     = 10   # full checkpoint cadence (epochs)


# Dataset
class CXRPretrainDataset(Dataset):
    """
    Unlabeled CXR dataset for SimMIM pretraining.

    Uses the same CLAHE + per-image standardization pipeline as the supervised
    train.py so that pretrained features transfer without a distribution shift.
    Augmentations are conservative and anatomically appropriate for chest X-rays:
      - Horizontal flip only  (left-right symmetry is valid for CXRs)
      - Small rotation        (±10°)
      - No vertical flip      (anatomically wrong)
      - No color jitter       (images are converted to grayscale→RGB)
    """
    def __init__(self, root: str, img_size: int = 256):
        self.paths = sorted(
            glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
            + glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)
        )
        if not self.paths:
            raise RuntimeError(f"No images found under {root}")

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            CLAHETransform(clip_limit=CLIP_LIMIT, tile_grid_size=TILE_GRID_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.5),
            transforms.ToTensor(),
            PerImageStandardize(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img)


# FPN Decoder
class FPNDecoder(nn.Module):
    """
    Lightweight FPN that fuses all 4 backbone stage outputs and progressively
    upsamples to the patch grid (64×64 for 256px input / patch_size=4).

    Stage output spatial sizes (SwinV2-Small, 256px, patch_size=4):
        stage[0]:  32×32   C=192
        stage[1]:  16×16   C=384
        stage[2]:   8×8    C=768
        stage[3]:   8×8    C=768  ← deepest; norm applied in _encode

    Decoder path:
        8×8   fuse stage[2] + stage[3]
        16×16  upsample 2× + fuse stage[1]
        32×32  upsample 2× + fuse stage[0]
        64×64  upsample 2× + refine
        head → (in_chans × patch_size², 64, 64)

    Each lateral branch has its own LayerNorm so intermediate encoder features
    (which lack a backbone norm) are normalized before projection.
    """

    def __init__(
        self,
        encoder_dims: list,
        decoder_dim: int,
        out_channels: int,
        stage_hw: list,
    ):
        super().__init__()
        self.stage_hw = stage_hw   # [(32,32),(16,16),(8,8),(8,8)]

        # Lateral: token-space LayerNorm + linear projection to decoder_dim
        self.lat = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(enc_dim),
                nn.Linear(enc_dim, decoder_dim),
            )
            for enc_dim in encoder_dims
        ])

        def fuse(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.GELU(),
            )

        D = decoder_dim
        self.fuse_8  = fuse(D * 2, D)   # merge stage[2] and stage[3]
        self.fuse_16 = fuse(D * 2, D)   # merged + stage[1] skip
        self.fuse_32 = fuse(D * 2, D)   # merged + stage[0] skip
        self.fuse_64 = fuse(D,     D)   # final refinement before head

        self.head = nn.Conv2d(D, out_channels, kernel_size=1)

    @staticmethod
    def _to_feat(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """(B, h*w, C) → (B, C, h, w)"""
        B, _, C = tokens.shape
        return tokens.transpose(1, 2).reshape(B, C, h, w)

    def forward(self, stage_feats: list) -> torch.Tensor:
        # Project each stage to decoder_dim, reshape to spatial feature maps
        fmaps = [
            self._to_feat(self.lat[i](stage_feats[i]), *self.stage_hw[i])
            for i in range(4)
        ]

        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)

        x = self.fuse_8(torch.cat([fmaps[2], fmaps[3]], dim=1))   # 8×8
        x = self.fuse_16(torch.cat([up(x),   fmaps[1]], dim=1))   # 16×16
        x = self.fuse_32(torch.cat([up(x),   fmaps[0]], dim=1))   # 32×32
        x = self.fuse_64(up(x))                                    # 64×64
        return self.head(x)                                        # (B, P, 64, 64)


# SimMIM model
class SimMIM_SwinV2(nn.Module):
    """
    SimMIM pretraining wrapper around SwinTransformerV2.

    Key design choices
    
    Mask token
        A shared learned parameter replaces masked patch embeddings (SimMIM 3.1).
        Zero-fill (original bug) encodes position information through the specific
        value 0.0 in embedding space, undermining masked modelling.

    Reconstruction target
        Per-patch L1 loss on *normalized* pixel patches (mean=0, std=1 per patch).
        Normalization removes global CXR intensity variation and focuses the
        objective on local texture — the signal most useful for pathology detection.

    Decoder
        FPN with skip connections from all 4 encoder stages.  The original bilinear
        8×→64× upsample discards three quarters of the spatial hierarchy.
    """

    def __init__(
        self,
        backbone: SwinTransformerV2,
        img_size: int   = 256,
        patch_size: int = 4,
        in_chans: int   = 3,
        mask_ratio: float = 0.6,
        decoder_dim: int  = 256,
    ):
        super().__init__()
        self.backbone   = backbone
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

        H0, W0 = backbone.patches_resolution          # 64, 64
        self.patch_h     = H0
        self.patch_w     = W0
        self.num_patches = H0 * W0                    # 4096

        # Learned mask token
        C0 = backbone.embed_dim                       # 96
        self.mask_token = nn.Parameter(torch.zeros(1, 1, C0))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Derive per-stage output shapes from backbone
        # PatchMerging fires at the END of each layer except the last,
        # halving H and W and doubling C.
        enc_dims, stage_hw = [], []
        H, W = H0, W0
        for i, layer in enumerate(backbone.layers):
            has_down = (layer.downsample is not None)
            if has_down:
                H, W = H // 2, W // 2
            stage_hw.append((H, W))
            # Output channel dim = embed_dim * 2^min(i+1, num_layers-1)
            enc_dims.append(
                backbone.embed_dim
                * (2 ** min(i + 1, backbone.num_layers - 1))
            )
        # stage_hw  = [(32,32), (16,16), (8,8), (8,8)]
        # enc_dims  = [192,     384,     768,   768  ]

        # FPN decoder
        self.decoder = FPNDecoder(
            encoder_dims=enc_dims,
            decoder_dim=decoder_dim,
            out_channels=in_chans * patch_size * patch_size,
            stage_hw=stage_hw,
        )

    # Masking
    def _random_mask(self, B: int, L: int, device) -> torch.Tensor:
        """Vectorized random mask: True = masked.  Exactly round(L*ratio) per row."""
        num_mask = int(round(L * self.mask_ratio))
        ids      = torch.argsort(torch.rand(B, L, device=device), dim=1)
        mask     = torch.zeros(B, L, dtype=torch.bool, device=device)
        mask.scatter_(1, ids[:, :num_mask], True)
        return mask

    # Encoder (manual forward to capture stage outputs)
    def _encode(self, imgs: torch.Tensor, mask: torch.Tensor) -> list:
        x = self.backbone.patch_embed(imgs)           # (B, L, C0)
        B, L, C0 = x.shape

        # Replace masked positions with the learned mask token
        mt = self.mask_token.expand(B, L, -1)
        x  = torch.where(mask.unsqueeze(-1), mt, x)

        if self.backbone.ape:
            x = x + self.backbone.absolute_pos_embed
        x = self.backbone.pos_drop(x)

        # Run each stage and collect output tokens
        # backbone.norm is NOT applied here; the decoder's lateral LayerNorms
        # handle normalization uniformly across all stages.
        stage_feats = []
        for layer in self.backbone.layers:
            x = layer(x)
            stage_feats.append(x)

        return stage_feats

    # Forward
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        B = imgs.size(0)
        device = imgs.device

        # Random binary patch mask
        mask = self._random_mask(B, self.num_patches, device)   # (B, L)

        # Masked encoding — collect all stage feature maps
        stage_feats = self._encode(imgs, mask)

        # FPN decode → predicted pixel content
        pred_map = self.decoder(stage_feats)                    # (B, P, 64, 64)
        pred     = pred_map.flatten(2).transpose(1, 2)          # (B, L, P)

        # Pixel-patch targets via unfold
        targets = F.unfold(
            imgs,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2)                                       # (B, L, P)

        # Per-patch target normalization
        #    Removes global intensity variation common in CXRs.
        #    Focuses the reconstruction objective on local structure (edges,
        #    vessels, infiltrates) rather than absolute brightness — exactly
        #    the features relevant for pathology classification.
        t_mean = targets.mean(dim=-1, keepdim=True)
        t_std  = targets.std(dim=-1, keepdim=True).clamp(min=1e-6)
        targets = (targets - t_mean) / t_std

        # L1 loss on masked patches only
        loss = F.l1_loss(pred[mask], targets[mask])
        return loss


# LR schedule
def get_lr(base_lr: float, epoch: int, num_epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    t = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


# Checkpoint helpers
def save_full_ckpt(path: str, epoch: int, model, optimizer, scaler, best_loss: float):
    """Full training state — sufficient to resume exactly."""
    raw = model.module if isinstance(model, DDP) else model
    torch.save({
        "epoch":     epoch,
        "model":     raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
        "best_loss": best_loss,
    }, path)


def save_backbone_ckpt(path: str, model):
    """
    Backbone-only checkpoint in the format expected by train.py init_ckpt:

        {"model": {"encoder.<key>": <tensor>, ...}}

    init_ckpt strips the "encoder." prefix and loads directly into
    model.backbone, so this wrapper is the only required adapter.
    """
    raw = model.module if isinstance(model, DDP) else model
    backbone_sd = raw.backbone.state_dict()
    torch.save(
        {"model": {"encoder." + k: v for k, v in backbone_sd.items()}},
        path,
    )


# Main
def main():
    # DDP init
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device     = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        is_main    = (local_rank == 0)
        world_size = dist.get_world_size()
    else:
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main    = True
        local_rank = 0
        world_size = 1

    torch.backends.cudnn.benchmark = True

    if is_main:
        print(f"[{datetime.datetime.now()}]  SimMIM CXR pretraining")
        print(f"  GPUs: {world_size}  |  batch/GPU: {BATCH_SIZE}  |  "
              f"effective batch: {BATCH_SIZE * world_size * ACCUM_STEPS}")

    # Dataset & loader
    ds      = CXRPretrainDataset(IMAGE_ROOT, img_size=IMG_SIZE)
    sampler = DistributedSampler(ds, shuffle=True) if ddp else None

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    if is_main:
        print(f"  Dataset: {len(ds)} images  |  steps/epoch: {len(loader)}")

    # Model
    backbone = SwinTransformerV2(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        embed_dim=EMBED_DIM,
        depths=DEPTHS,
        num_heads=NUM_HEADS,
        window_size=WINDOW_SIZE,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
    )

    model = SimMIM_SwinV2(
        backbone=backbone,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        mask_ratio=MASK_RATIO,
        decoder_dim=DECODER_DIM,
    ).to(device)

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # Optimizer & scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Resume
    start_epoch = 0
    best_loss   = float("inf")

    if RESUME_CKPT and os.path.isfile(RESUME_CKPT):
        ckpt    = torch.load(RESUME_CKPT, map_location="cpu", weights_only=True)
        raw     = model.module if isinstance(model, DDP) else model
        raw.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss   = ckpt.get("best_loss", float("inf"))
        if is_main:
            print(f"  Resumed from epoch {ckpt['epoch']}  (best_loss={best_loss:.4f})")

    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        if ddp:
            sampler.set_epoch(epoch)

        model.train()
        lr = get_lr(BASE_LR, epoch, NUM_EPOCHS, WARMUP_EPOCHS)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        epoch_total_loss = 0.0
        epoch_steps = 0
        running_loss   = 0.0
        accum_count    = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", disable=not is_main)
        for it, imgs in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                loss = model(imgs) / ACCUM_STEPS

            scaler.scale(loss).backward()
            accum_count += 1

            if accum_count == ACCUM_STEPS:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0

            running_loss += loss.item() * ACCUM_STEPS
            epoch_total_loss += loss.item() * ACCUM_STEPS
            epoch_steps += 1

            if is_main and (it + 1) % PRINT_FREQ == 0:
                avg = running_loss / PRINT_FREQ
                pbar.write(
                    f"  [Ep {epoch + 1:03d}  it {it + 1:5d}] "
                    f"lr={lr:.2e}  loss={avg:.4f}"
                )
                if avg < best_loss:
                    best_loss = avg
                running_loss = 0.0

        # Flush any leftover accumulated gradients at epoch end
        if accum_count > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)


        # Periodic saves (rank 0 only)
        if is_main and (epoch + 1) % SAVE_EVERY == 0:
            epoch_loss = epoch_total_loss / max(epoch_steps, 1)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
            full_path = f"simmim_full_epoch{epoch + 1:03d}.pth"
            bk_path   = f"simmim_backbone_epoch{epoch + 1:03d}.pth"
            save_full_ckpt(full_path, epoch, model, optimizer, scaler, best_loss)
            save_backbone_ckpt(bk_path, model)
            print(f"  → saved {full_path}  +  {bk_path}")

    # Final backbone save 
    if is_main:
        save_backbone_ckpt(OUTPUT_CKPT, model)
        print(f"\nDone. Backbone checkpoint → {OUTPUT_CKPT}")
        print(f"Best observed loss: {best_loss:.4f}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()