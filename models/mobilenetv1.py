"""
MobileNetV1 implementation in the style of
<Knowledge Distillation via Route Constrained Optimization>
"""

import torch
import torch.nn as nn
import math

__all__ = ['mobilenetv1_w', 'mobilev1_half']

BN = None


def conv_dw(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class MobileNetV1(nn.Module):
    def __init__(self,
                 feature_dim=100,
                 width_mult=1.0,
                 input_size=32,
                 remove_avg=False):
        super(MobileNetV1, self).__init__()
        self.remove_avg = remove_avg
        input_channel = int(32 * width_mult)

        def c(ch): return int(ch * width_mult)

        self.features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU(inplace=True)
            ),
            conv_dw(input_channel, c(64), 1),
            conv_dw(c(64), c(128), 2),
            conv_dw(c(128), c(128), 1),
            conv_dw(c(128), c(256), 2),
            conv_dw(c(256), c(256), 1),
            conv_dw(c(256), c(512), 2),
            nn.Sequential(*[conv_dw(c(512), c(512), 1) for _ in range(5)]),
            conv_dw(c(512), c(1024), 2),
            conv_dw(c(1024), c(1024), 1)
        ])

        self.last_channel = c(1024)
        self.avgpool = nn.AvgPool2d(input_size // 32, ceil_mode=True)

        self.classifier = nn.Linear(self.last_channel, feature_dim)

        self._initialize_weights()
        print(width_mult)

    def get_bn_before_relu(self):
        # Return batchnorms before relu of selected layers
        return [
            self.features[2][1],  # after conv_dw_2
            self.features[4][1],  # after conv_dw_4
            self.features[6][1],  # after conv_dw_6
            self.features[9][1],  # after last conv_dw
        ]

    def get_feat_modules(self):
        return self.features

    def forward(self, x, is_feat=False, preact=False):
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [0, 2, 4, 6, 9]:  # record key feature maps
                feats.append(x)

        if not self.remove_avg:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feats.append(x)

        out = self.classifier(x)

        if is_feat:
            return feats, out
        else:
            return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv1_w(W, feature_dim=100):
    return MobileNetV1(width_mult=W, feature_dim=feature_dim)


def mobilev1_half(num_classes):
    return mobilenetv1_w(0.5, num_classes)


if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)
    net = mobilev1_half(100)

    feats, logit = net(x, is_feat=True)
    for f in feats:
        print(f.shape)
    print(logit.shape)

    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')
        else:
            print('warning')
