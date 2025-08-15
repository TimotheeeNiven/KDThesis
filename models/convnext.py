import torch
import torch.nn as nn
import timm
from timm.models import create_model

__all__ = ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large']


class ConvNeXtKDWrapper(nn.Module):
    def __init__(self, model_name='convnext_large', num_classes=100, proj_dim=128, pretrained=True):
        super().__init__()
        self.model = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            features_only=True  # Extract intermediate features from stages
        )
        feature_dim = self.model.feature_info[-1]['num_chs']
        self.norm = nn.LayerNorm(feature_dim)
        self.proj_head = nn.Linear(feature_dim, proj_dim)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x, is_feat=False, preact=False):
        feats = self.model(x)                    # List of feature maps from each stage
        x = feats[-1].mean([-2, -1])             # Global average pooling on final feature map
        x = self.norm(x)
        proj = self.proj_head(x)
        out = self.head(x)
        return (feats + [proj], out) if is_feat else out


# === Factory Functions ===
def convnext_tiny(pretrained=False, num_classes=100, proj_dim=128, **kwargs):
    return ConvNeXtKDWrapper('convnext_tiny', num_classes, proj_dim, pretrained=pretrained)


def convnext_small(pretrained=False, num_classes=100, proj_dim=128, **kwargs):
    return ConvNeXtKDWrapper('convnext_small', num_classes, proj_dim, pretrained=pretrained)


def convnext_base(pretrained=True, num_classes=100, proj_dim=128, **kwargs):
    return ConvNeXtKDWrapper('convnext_base', num_classes, proj_dim, pretrained=pretrained)


def convnext_large(pretrained=True, num_classes=100, proj_dim=128, **kwargs):
    return ConvNeXtKDWrapper('convnext_large', num_classes, proj_dim, pretrained=pretrained)


# === Test ===
if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    model = convnext_large(pretrained=True)
    feats, out = model(x, is_feat=True)
    print("Feature shapes:", [f.shape for f in feats])
    print("Logits shape:", out.shape)
