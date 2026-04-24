import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch

from dataset import ALL_CLASSES
from train import get_class_attention



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


class GradCAM:
    """
    GradCAM for the SwinWithView model.
    Hooks the final backbone stage output (patch tokens before norm).
    """
    def __init__(self, model, device):
        self.model  = model
        self.device = device
        self._feats = None
        self._grads = None

        # Hook the output of the last backbone stage
        target = model.backbone.layers[-1]
        self._fwd_hook = target.register_forward_hook(self._save_feats)
        self._bwd_hook = target.register_full_backward_hook(self._save_grads)

    def _save_feats(self, module, input, output):
        # output may be a tuple depending on SwinV2 implementation
        self._feats = output[0] if isinstance(output, tuple) else output

    def _save_grads(self, module, grad_input, grad_output):
        self._grads = grad_output[0] if isinstance(grad_output, tuple) else grad_output[0]

    def __call__(self, img_tensor, class_idx, view_id=0):
        self.model.eval()
        x = img_tensor.unsqueeze(0).to(self.device).requires_grad_(False)
        v = torch.tensor([view_id], dtype=torch.long, device=self.device)

        logits = self.model(x, v)               # (1, num_classes)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # feats / grads: (1, N, C)
        grads = self._grads.detach()            # (1, N, C)
        feats = self._feats.detach()            # (1, N, C)

        weights = grads.mean(dim=-1, keepdim=True)  # (1, N, 1)  -- GAP over C
        cam = (weights * feats).sum(dim=-1)     # (1, N)
        cam = cam.squeeze(0)                    # (N,)
        cam = torch.relu(cam)

        H = W = int(cam.shape[0] ** 0.5)
        cam = cam.reshape(H, W).cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def overlay_heatmap(img_np, heatmap_hw, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    img_np    : (H, W) or (H, W, 3) float32 in [0,1] or uint8
    heatmap_hw: (h, w) float in [0,1]
    Returns   : (H, W, 3) uint8 BGR
    """
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    H, W = img_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap_hw, (W, H), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8   = (heatmap_resized * 255).clip(0, 255).astype(np.uint8)
    colored         = cv2.applyColorMap(heatmap_uint8, colormap)
    return cv2.addWeighted(img_np, 1 - alpha, colored, alpha, 0)


def visualize_class(model, img_tensor, img_np, class_idx, device,
                    gradcam: GradCAM, view_id=0, save_path=None):
    """
    Side-by-side: original | GradCAM | attention map
    img_tensor : (3, H, W) torch tensor (normalised)
    img_np     : (H, W) or (H, W, 3) raw image for display
    """
    class_name = ALL_CLASSES[class_idx]

    # Attention map (no grad needed)
    attn_map = get_class_attention(model, img_tensor, class_idx, device)  # (h, w)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() + 1e-8)

    # GradCAM
    cam = gradcam(img_tensor, class_idx, view_id=view_id)

    # Overlays
    attn_overlay = overlay_heatmap(img_np.copy(), attn_map,
                                   colormap=cv2.COLORMAP_INFERNO)
    cam_overlay  = overlay_heatmap(img_np.copy(), cam,
                                   colormap=cv2.COLORMAP_JET)

    # Convert BGR -> RGB for matplotlib
    attn_rgb = cv2.cvtColor(attn_overlay, cv2.COLOR_BGR2RGB)
    cam_rgb  = cv2.cvtColor(cam_overlay,  cv2.COLOR_BGR2RGB)

    if img_np.ndim == 2:
        orig_disp = img_np
        orig_cmap = "gray"
    else:
        orig_disp = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_BGR2RGB)
        orig_cmap = None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Class: {class_name}", fontsize=14)

    axes[0].imshow(orig_disp, cmap=orig_cmap); axes[0].set_title("Original");     axes[0].axis("off")
    axes[1].imshow(cam_rgb);                   axes[1].set_title("GradCAM");       axes[1].axis("off")
    axes[2].imshow(attn_rgb);                  axes[2].set_title("Attention Map"); axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


