from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['resnext']

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 grouped convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)

class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None, is_last=False):
        super(ResNeXtBottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        preact = out
        out = F.relu(out)
        
        if self.is_last:
            return out, preact
        else:
            return out

class ResNeXt(nn.Module):
    def __init__(self, depth, num_filters, cardinality=32, num_classes=10):
        super(ResNeXt, self).__init__()
        assert (depth - 2) % 9 == 0, 'ResNeXt depth should be 9n+2, e.g. 50, 101, 152'
        n = (depth - 2) // 9

        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(ResNeXtBottleneck, num_filters[1], n, cardinality)
        self.layer2 = self._make_layer(ResNeXtBottleneck, num_filters[2], n, cardinality, stride=2)
        self.layer3 = self._make_layer(ResNeXtBottleneck, num_filters[3], n, cardinality, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(num_filters[3] * ResNeXtBottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cardinality, stride, downsample, is_last=(blocks == 1)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality, is_last=(i == blocks-1)))

        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, preact=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = x

        x, f1_pre = self.layer1(x)
        f1 = x
        x, f2_pre = self.layer2(x)
        f2 = x
        x, f3_pre = self.layer3(x)
        f3 = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f4 = x
        x = self.fc(x)

        if is_feat:
            if preact:
                return [f0, f1_pre, f2_pre, f3_pre, f4], x
            else:
                return [f0, f1, f2, f3, f4], x
        else:
            return x


def resnext50_32x4d(**kwargs):
    return ResNeXt(50, [64, 128, 256, 512], cardinality=32, **kwargs)

def resnext101_32x8d(**kwargs):
    return ResNeXt(101, [64, 256, 512, 1024], cardinality=32, **kwargs)

def resnext152_32x8d(**kwargs):
    return ResNeXt(152, [64, 256, 512, 1024], cardinality=32, **kwargs)

if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 32, 32)
    net = resnext101_32x8d(num_classes=20)
    feats, logit = net(x, is_feat=True, preact=True)

    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)