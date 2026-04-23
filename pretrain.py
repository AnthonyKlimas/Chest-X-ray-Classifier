
import os
import glob
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from swin_transformer_v2 import SwinTransformerV2


### Path Configuration ###
IMAGE_ROOT = "../chest_xray_dataset/CXR8/images_preprocessed"
OUTPUT_CKPT = "../chest_xray_dataset/simmim_swinv2_cxr_backbone.pth"


### SimMIM Parameters ###
IMG_SIZE = 256
PATCH_SIZE = 4
IN_CHANS = 3

BATCH_SIZE = 8
ACCUM_STEPS = 4
NUM_EPOCHS = 50
MASK_RATIO = 0.6

BASE_LR = 1e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 5
NUM_WORKERS = 10
PRINT_FREQ = 180
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Unlabeled CXR dataset
class UnlabeledCXRDataset(Dataset):
    def __init__(self, root, img_size=256):
        self.paths = sorted(
            glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
            + glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)
            + glob.glob(os.path.join(root, "**", "*.jpeg"), recursive=True)
        )
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under {root}")

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # [0,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        img = self.tf(img)
        return img


# SimMIM for SwinV2 (correct)
class SimMIM_SwinV2(nn.Module):
    """
    SimMIM-style pretraining for hierarchical SwinV2:
      - mask at patch level (64x64 = 4096 patches)
      - encode with full Swin
      - upsample encoder tokens back to patch grid
      - reconstruct pixel patches at masked positions
    """
    def __init__(self, backbone: SwinTransformerV2,
                 img_size=256, patch_size=4, in_chans=3, mask_ratio=0.6):
        super().__init__()
        self.backbone = backbone
        self.mask_ratio = mask_ratio

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        # Patch grid
        self.num_patches_h = img_size // patch_size
        self.num_patches_w = img_size // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w  # 64*64=4096

        # Final encoder dim (SwinV2-Small: 768)
        self.encoder_dim = backbone.num_features

        # Decoder: upsample from final feature map back to patch grid
        # Final tokens are 8x8 (for 256/4/2/2/2), we upsample to 64x64
        self.decoder = nn.Sequential(
            nn.Conv2d(self.encoder_dim, self.encoder_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.encoder_dim, in_chans * patch_size * patch_size, kernel_size=1),
        )

    def _random_mask(self, B, L, device):
        """
        Generate a random boolean mask of shape (B, L) with mask_ratio.
        True = masked, False = visible.
        """
        num_mask = int(L * self.mask_ratio)
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for i in range(B):
            idx = torch.randperm(L, device=device)[:num_mask]
            mask[i, idx] = True
        return mask

    def forward(self, imgs):
        """
        imgs: (B, 3, H, W)
        Returns: scalar loss
        """
        B = imgs.size(0)
        device = imgs.device

        # Patchify (before Swin layers)
        x = self.backbone.patch_embed(imgs)  # (B, L, C0)
        B_, L, C0 = x.shape
        assert L == self.num_patches, f"Expected {self.num_patches} patches, got {L}"

        # Create mask at patch level
        mask = self._random_mask(B, L, device=device)  # (B, L)
        x_masked = x.clone()
        x_masked[mask] = 0.0

        # Run Swin encoder manually (no avgpool)
        if self.backbone.ape:
            x_masked = x_masked + self.backbone.absolute_pos_embed
        x_masked = self.backbone.pos_drop(x_masked)

        for layer in self.backbone.layers:
            x_masked = layer(x_masked)

        x_masked = self.backbone.norm(x_masked)  # (B, L_enc, C_enc)

        # For SwinV2-Small with 256x256 and patch_size=4:
        # L_enc = 4*4 = 16 tokens (final stage), but we know spatial shape:
        # patches_resolution = [64, 64]
        # final stage resolution = 64 / 8 = 8, so 8x8=64 tokens.
        # However, SwinV2 here uses 4 stages with PatchMerging, so:
        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4 (depends on depths).
        # Let's compute it from backbone.patches_resolution and num_layers.

        H0, W0 = self.backbone.patches_resolution  # e.g. [64, 64]
        # After num_layers-1 PatchMerging operations:
        H_enc = H0 // (2 ** (self.backbone.num_layers - 1))
        W_enc = W0 // (2 ** (self.backbone.num_layers - 1))
        # So L_enc should be H_enc * W_enc
        assert x_masked.shape[1] == H_enc * W_enc, \
            f"Encoder tokens {x_masked.shape[1]} != {H_enc*W_enc}"

        # Reshape encoder tokens to feature map
        x_feat = x_masked.transpose(1, 2).contiguous()  # (B, C_enc, L_enc)
        x_feat = x_feat.view(B, self.encoder_dim, H_enc, W_enc)  # (B, C_enc, H_enc, W_enc)

        # Upsample to patch grid resolution (64x64)
        x_up = F.interpolate(
            x_feat,
            size=(self.num_patches_h, self.num_patches_w),
            mode="bilinear",
            align_corners=False,
        )  # (B, C_enc, 64, 64)

        # Predict pixel patches
        pred_map = self.decoder(x_up)  # (B, P, 64, 64), P = in_chans * patch_size^2

        # Convert pred_map to (B, L, P)
        pred = pred_map.flatten(2).transpose(1, 2)  # (B, L, P)

        # Build pixel patch targets from original images
        patches = F.unfold(
            imgs,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )  # (B, P, L)
        patches = patches.transpose(1, 2)  # (B, L, P)

        # Compute loss only on masked patches
        # mask: (B, L) boolean
        pred_masked = pred[mask].view(-1, patches.size(-1))      # (N_masked, P)
        target_masked = patches[mask].view(-1, patches.size(-1)) # (N_masked, P)


        loss = F.l1_loss(pred_masked, target_masked)
        return loss


# LR schedule (cosine + warmup)
def cosine_lr_schedule(base_lr, epoch, num_epochs, warmup_epochs):
    if epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(warmup_epochs)
    t = (epoch - warmup_epochs) / max(1, (num_epochs - warmup_epochs))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


# Main training loop
def main():
    torch.backends.cudnn.benchmark = True

    ds = UnlabeledCXRDataset(IMAGE_ROOT, img_size=IMG_SIZE)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    backbone = SwinTransformerV2(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
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
        use_checkpoint=True,  # save VRAM on 3070
    )

    model = SimMIM_SwinV2(
        backbone=backbone,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        mask_ratio=MASK_RATIO,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    global_step = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        lr = cosine_lr_schedule(BASE_LR, epoch, NUM_EPOCHS, WARMUP_EPOCHS)
        for g in optimizer.param_groups:
            g["lr"] = lr

        running_loss = 0.0
        for it, imgs in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            imgs = imgs.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda"), dtype=torch.bfloat16):
                loss = model(imgs) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (it + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running_loss += loss.item() * ACCUM_STEPS

            if (it + 1) % PRINT_FREQ == 0:
                avg_loss = running_loss / PRINT_FREQ
                print(f"[Epoch {epoch+1} Iter {it+1}] lr={lr:.2e} loss={avg_loss:.4f}")
                running_loss = 0.0

        # Save backbone checkpoint every few epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == NUM_EPOCHS:
            ckpt_path = f"simmim_swinv2_cxr_backbone_epoch{epoch+1:03d}.pth"
            torch.save(model.backbone.state_dict(), ckpt_path)
            print(f"Saved backbone checkpoint to {ckpt_path}")

    torch.save(model.backbone.state_dict(), OUTPUT_CKPT)
    print(f"Done. Final backbone saved to {OUTPUT_CKPT}")


if __name__ == "__main__":
    main()
