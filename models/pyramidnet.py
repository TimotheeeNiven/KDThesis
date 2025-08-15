from __future__ import absolute_import

"""PyramidNet for CIFAR datasets.
Fully custom-built version without external model dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['pyramidnet272', 'pyramidnet110']


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)
        return out


class PyramidNet(nn.Module):
    def __init__(self, depth, alpha, num_classes=10):
        super(PyramidNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'Depth must be 6n+2 for BasicBlock PyramidNet.'

        self.inplanes = 16
        n = (depth - 2) // 6
        self.addrate = alpha / (3 * n)
        self.featuremap_dim = float(self.inplanes)

        self.input_featuremap_dim = self.inplanes
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, n, stride=1)
        self.layer2 = self._make_layer(BasicBlock, n, stride=2)
        self.layer3 = self._make_layer(BasicBlock, n, stride=2)

        self.final_featuremap_dim = int(round(self.featuremap_dim))
        self.bn_final = nn.BatchNorm2d(self.final_featuremap_dim)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

        self._init_weights()

    def _make_layer(self, block, blocks, stride):
        layers = []
        for i in range(blocks):
            self.featuremap_dim += self.addrate
            outplanes = int(round(self.featuremap_dim))
            current_stride = stride if i == 0 else 1

            if current_stride != 1 or self.input_featuremap_dim != outplanes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.input_featuremap_dim, outplanes * block.expansion,
                              kernel_size=1, stride=current_stride, bias=False),
                    nn.BatchNorm2d(outplanes * block.expansion)
                )
            else:
                downsample = None

            layers.append(block(self.input_featuremap_dim, outplanes, current_stride, downsample))
            self.input_featuremap_dim = outplanes * block.expansion

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, is_feat=False, preact=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = x

        x = self.layer1(x)
        f1 = f1_pre = x
        x = self.layer2(x)
        f2 = f2_pre = x
        x = self.layer3(x)
        f3 = f3_pre = x

        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        f4 = x
        x = self.fc(x)

        if is_feat:
            return ([f0, f1_pre, f2_pre, f3_pre, f4], x) if preact else ([f0, f1, f2, f3, f4], x)
        else:
            return x

    def get_feat_modules(self):
        return nn.ModuleList([
            self.conv1,
            self.bn1,
            self.relu,
            self.layer1,
            self.layer2,
            self.layer3
        ])

    def get_bn_before_relu(self):
        return [
            self.layer1[-1].bn2,
            self.layer2[-1].bn2,
            self.layer3[-1].bn2
        ]


def pyramidnet272(**kwargs):
    return PyramidNet(depth=272, alpha=150, **kwargs)

def pyramidnet110(**kwargs):
    return PyramidNet(depth=110, alpha=84, **kwargs)


if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)
    model = pyramidnet110(num_classes=100)
    model.eval()
    feats, out = model(x, is_feat=True, preact=True)

    for f in feats:
        print(f.shape, f.min().item())
    print(out.shape)

    for bn in model.get_bn_before_relu():
        print('pass' if isinstance(bn, nn.BatchNorm2d) else 'warning')
