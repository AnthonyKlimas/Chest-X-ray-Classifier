import torch
import torch.nn as nn
from timm.layers import DropPath, to_2tuple, trunc_normal_

# Minimal SwinTransformerV2 backbone for loading SimMIM weights
class SwinTransformerV2(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24],
                 window_size=8, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1):
        super().__init__()

        from timm.models.swin_transformer_v2 import SwinTransformerV2 as TimmSwin

        self.model = TimmSwin(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            num_classes=0,
            pretrained=False
        )

    # REQUIRED for your SwinWithView wrapper
    def forward_features(self, x):
        return self.model.forward_features(x)

    # Optional: forward() just calls forward_features()
    def forward(self, x):
        return self.forward_features(x)

    @property
    def head(self):
        return self.model.head

    @head.setter
    def head(self, v):
        self.model.head = v

    @property
    def layers(self):
        return self.model.layers
