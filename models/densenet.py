from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F
import torch
import math

__all__ = ['densenet121', 'densenet169']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super(DenseLayer, self).__init__()
        inter_channels = bn_size * growth_rate
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = conv3x3(inter_channels, growth_rate)

        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(F.relu(self.norm1(x)))
        out = self.conv2(F.relu(self.norm2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.norm(x))))

class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate, init_channels=64, bn_size=4, drop_rate=0, num_classes=100):
        super(DenseNet, self).__init__()
        self.in_channels = init_channels

        self.conv1 = nn.Conv2d(3, init_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(init_channels)
        self.relu = nn.ReLU(inplace=True)

        self.features = []
        for num_layers in num_blocks:
            self.features.append(DenseBlock(num_layers, self.in_channels, growth_rate, bn_size, drop_rate))
            self.in_channels += num_layers * growth_rate
            if num_layers != num_blocks[-1]:
                self.features.append(TransitionLayer(self.in_channels, self.in_channels // 2))
                self.in_channels //= 2
        self.features = nn.Sequential(*self.features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, is_feat=False, preact=False):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        f0 = x

        feature_maps = []
        for layer in self.features:
            x = layer(x)
            feature_maps.append(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        f5 = x
        x = self.fc(x)

        if is_feat:
            return [f0] + feature_maps + [f5], x
        else:
            return x

def densenet121(**kwargs):
    return DenseNet(
        num_blocks=[6, 12, 24, 16],
        growth_rate=32,
        init_channels=64,
        **kwargs
    )

def densenet169(**kwargs):
    return DenseNet(
        num_blocks=[6, 12, 32, 32],
        growth_rate=32,
        init_channels=64,
        **kwargs
    )

if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    net = densenet121(num_classes=100)
    feats, logit = net(x, is_feat=True)
    
    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)
