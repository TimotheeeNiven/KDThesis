from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F
import torch
import math

__all__ = ['regnety_16gf','regnety_4gf','regnety_32gf']

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        out = self.avgpool(x).view(batch, channels)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).view(batch, channels, 1, 1)
        return x * out

class RegNetYBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, se_ratio=0.25):
        super(RegNetYBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(planes * self.expansion, reduction=int(1 / se_ratio))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return F.relu(out)

class RegNetY(nn.Module):
    def __init__(self, depths, widths, groups, num_classes=10):
        super(RegNetY, self).__init__()
        self.inplanes = widths[0]
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(RegNetYBlock, widths[1], depths[0], groups)
        self.layer2 = self._make_layer(RegNetYBlock, widths[2], depths[1], groups, stride=2)
        self.layer3 = self._make_layer(RegNetYBlock, widths[3], depths[2], groups, stride=2)
        self.layer4 = self._make_layer(RegNetYBlock, widths[4], depths[3], groups, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[4], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, groups, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups))

        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, preact=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = x

        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x)
        f4 = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        f5 = x
        x = self.fc(x)

        if is_feat:
            return [f0, f1, f2, f3, f4, f5], x
        else:
            return x
def regnety_32gf(**kwargs):
    return RegNetY(
        depths=[4, 8, 16, 4],
        widths=[64, 128, 256, 512, 1280],  # Wider network
        groups=16,  # More groups for better feature extraction
        **kwargs
    )

def regnety_16gf(**kwargs):
    return RegNetY(
        depths=[2, 7, 14, 2],
        widths=[32, 64, 160, 384, 1088],
        groups=8,
        **kwargs
    )

def regnety_4gf(**kwargs):
    return RegNetY(
        depths=[2, 6, 12, 2],
        widths=[24, 48, 112, 256, 640],
        groups=8,
        **kwargs
    )

if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    net = regnety_16gf(num_classes=1000)
    feats, logit = net(x, is_feat=True)
    
    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)
