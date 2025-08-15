from __future__ import absolute_import

'''Vision Transformer (ViT) variants for CIFAR-100 and Tiny-ImageNet.
Includes: Custom ViTs + Pretrained ViT from Hugging Face.
'''
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

__all__ = [
    'vit_tiny', 'vit_small', 'vit_medium', 'vit_base', 'vit_distilled',
    'vit_pretrained', 'vit_pretrained_imagetiny'
]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = F.gelu

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = self.norm1(src + src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + src2)
        return src

class ViT(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, num_classes=100,
                 img_size=224, patch_size=16, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward=4 * d_model, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        # Initialize parameters
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, is_feat=False, preact=False):
        B = x.size(0)
        x = self.patch_embed(x)                  # [B, d_model, H', W']
        x = x.flatten(2).transpose(1, 2)         # [B, N, d_model]

        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat((cls_token, x), dim=1)          # [B, N+1, d_model]
        x = x + self.pos_embed[:, :x.size(1), :]      # Add positional encoding

        x = x.transpose(0, 1)                         # [N+1, B, d_model]
        features = []
        for layer in self.encoder_layers:
            x = layer(x)
            features.append(x.transpose(0, 1))  # Save as [B, N+1, d_model]

        x = x.transpose(0, 1)                         # [B, N+1, d_model]
        x = self.norm(x[:, 0])                        # CLS token
        out = self.head(x)

        if is_feat:
            return features, out
        else:
            return out

# === ViT Variants for 224x224 input ===

def vit_tiny(num_classes=100):
    return ViT(num_layers=4, d_model=192, num_heads=3, num_classes=num_classes,
               img_size=224, patch_size=16)

def vit_small(num_classes=100):
    return ViT(num_layers=6, d_model=384, num_heads=6, num_classes=num_classes,
               img_size=224, patch_size=16)

def vit_medium(num_classes=100):
    return ViT(num_layers=8, d_model=512, num_heads=8, num_classes=num_classes,
               img_size=224, patch_size=16)

def vit_base(num_classes=100):
    return ViT(num_layers=12, d_model=768, num_heads=12, num_classes=num_classes,
               img_size=224, patch_size=16)

def vit_distilled(num_classes=100):
    return ViT(num_layers=6, d_model=384, num_heads=6, num_classes=num_classes,
               img_size=224, patch_size=16, dropout=0.2)

# === Pretrained ViT fine-tuned on CIFAR-100 (via HuggingFace) ===

def vit_pretrained(num_classes=100):
    class WrappedViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = ViTForImageClassification.from_pretrained(
                'pkr7098/cifar100-vit-base-patch16-224-in21k'
            )

        def forward(self, x, is_feat=False, preact=False):
            out = self.vit(x).logits
            if is_feat:
                return [out], out  # Only logits available
            return out

    return WrappedViT()

# === Pretrained ViT from Hugging Face, re-headed for Tiny ImageNet ===

def vit_pretrained_imagetiny(num_classes):
    class TIMMViTWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = timm.create_model(
                'vit_base_patch16_224_miil',
                pretrained=True,
                num_classes=num_classes
            )

        def forward(self, x, is_feat=False, preact=False):
            out = self.vit(x)
            return ([out], out) if is_feat else out

    return TIMMViTWrapper()

# === Test block ===

if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    model = vit_tiny(num_classes=100)
    print("Custom Tiny ViT:", model(x, is_feat=True)[1].shape)

    pretrained_model = vit_pretrained()
    print("Best Pretrained ViT Output:", pretrained_model(x, is_feat=True)[1].shape)

    vit_tinyimg = vit_pretrained_imagetiny()
    print("HuggingFace TinyImageNet ViT Output:", vit_tinyimg(x, is_feat=True)[1].shape)
