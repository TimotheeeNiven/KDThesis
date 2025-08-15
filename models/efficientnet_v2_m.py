import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['efficientnet_v2_m']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class MBConv(nn.Module):
    """Mobile Inverted Residual Bottleneck Block"""
    def __init__(self, in_planes, out_planes, expand_ratio, stride, is_last=False):
        super(MBConv, self).__init__()
        self.is_last = is_last
        hidden_dim = in_planes * expand_ratio
        self.use_residual = stride == 1 and in_planes == out_planes

        self.conv1 = nn.Conv2d(in_planes, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.act2 = nn.SiLU()

        self.conv3 = nn.Conv2d(hidden_dim, out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_residual:
            out += residual

        preact = out  # Store pre-activation output
        out = F.silu(out)  # Apply SiLU activation

        return out, preact  # Always return both values

class EfficientNetV2M(nn.Module):
    """EfficientNetV2-M Model"""
    def __init__(self, width_mult=1.0, depth_mult=1.0, num_classes=100):
        super(EfficientNetV2M, self).__init__()

        self.stem = conv3x3(3, int(32 * width_mult), stride=1)
        self.bn1 = nn.BatchNorm2d(int(32 * width_mult))
        self.act1 = nn.SiLU()

        self.blocks = nn.Sequential(
            MBConv(int(32 * width_mult), int(24 * width_mult), expand_ratio=4, stride=2, is_last=False),
            MBConv(int(24 * width_mult), int(48 * width_mult), expand_ratio=4, stride=2, is_last=False),
            MBConv(int(48 * width_mult), int(64 * width_mult), expand_ratio=4, stride=2, is_last=False),
            MBConv(int(64 * width_mult), int(128 * width_mult), expand_ratio=6, stride=1, is_last=False),
            MBConv(int(128 * width_mult), int(160 * width_mult), expand_ratio=6, stride=2, is_last=False),
            MBConv(int(160 * width_mult), int(256 * width_mult), expand_ratio=6, stride=2, is_last=True),
        )

        self.head = nn.Conv2d(int(256 * width_mult), int(1280 * width_mult), kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(1280 * width_mult))
        self.act2 = nn.SiLU()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(1280 * width_mult), num_classes)

    def get_feat_modules(self):
        return nn.ModuleList([self.stem, self.bn1, self.act1, self.blocks])

    def get_bn_before_relu(self):
        return [self.bn1, self.bn2]

    def forward(self, x, is_feat=False, preact=False):
        x = self.stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        f0 = x

        f_list = []
        for i, block in enumerate(self.blocks):
            x, f_preact = block(x)
            f_list.append(x if not preact else f_preact)

        x = self.head(x)
        x = self.bn2(x)
        x = self.act2(x)
        f_list.append(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        f_list.append(x)
        x = self.fc(x)

        if is_feat:
            return f_list, x
        else:
            return x

def efficientnet_v2_m(num_classes=100):
    return EfficientNetV2M(num_classes=num_classes)

# Test
if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    net = efficientnet_v2_m()
    feats, logit = net(x, is_feat=True, preact=True)

    for f in feats:
        print(f.shape)
    print(logit.shape)
