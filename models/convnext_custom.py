# File: convnext_custom.py (finalized for RKD/Attention)
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'convnext_tiny_for_large',
    'convnext_tiny_for_base',
    'convnext_tiny_for_small'
]

class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXtStudent(nn.Module):
    def __init__(self, in_chans=3, num_classes=100, depths=[2, 2, 6, 2], dims=[96, 192, 384, 768]):
        super().__init__()
        self.depths = depths
        self.dims = dims

        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        for i in range(4):
            if i == 0:
                down = nn.Sequential(
                    nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                    LayerNorm2d(dims[0])
                )
            else:
                down = nn.Sequential(
                    LayerNorm2d(dims[i - 1]),
                    nn.Conv2d(dims[i - 1], dims[i], kernel_size=2, stride=2)
                )
            self.downsample_layers.append(down)

            stage_blocks = nn.ModuleList()
            for _ in range(depths[i]):
                block = ConvNeXtBlock(dim=dims[i])
                stage_blocks.append(block)
            self.stages.append(stage_blocks)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, is_feat=False, preact=None):
        feats = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            for j in range(self.depths[i]):
                x = self.stages[i][j](x)
            feats.append(x)

        x = x.mean([-2, -1])
        x = self.norm(x)
        out = self.head(x)

        return (feats, out) if is_feat else out


# === Factory Functions ===
def convnext_tiny_for_large(num_classes=100):
    return ConvNeXtStudent(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=num_classes)

def convnext_tiny_for_base(num_classes=100):
    return ConvNeXtStudent(depths=[2, 2, 6, 2], dims=[96, 192, 384, 768], num_classes=num_classes)

def convnext_tiny_for_small(num_classes=100):
    return ConvNeXtStudent(depths=[2, 2, 4, 1], dims=[64, 128, 256, 512], num_classes=num_classes)