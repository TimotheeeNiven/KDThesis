import torch
import torch.nn as nn
import math

__all__ = ['mobilenetv3_large', 'mobilenetv3_small','mobilenetv3_large_plus']


def conv_bn(inp, oup, kernel, stride, activation='relu'):
    act_layer = nn.ReLU(inplace=True) if activation == 'relu' else nn.Hardswish()
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, kernel // 2, bias=False),
        nn.BatchNorm2d(oup),
        act_layer
    )

def conv_1x1_bn(inp, oup, activation='relu'):
    act_layer = nn.ReLU(inplace=True) if activation == 'relu' else nn.Hardswish()
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        act_layer
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, kernel, stride, expand_ratio, activation, se=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = inp * expand_ratio
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = [
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.Hardswish(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel, stride, kernel // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.Hardswish(),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3(nn.Module):
    def __init__(self, cfgs, feature_dim, width_mult=1.0, input_size=32):
        super(MobileNetV3, self).__init__()
        self.cfgs = cfgs
        input_channel = int(16 * width_mult)
        self.conv1 = conv_bn(3, input_channel, 3, 1, activation='hardswish')
        self.blocks = nn.ModuleList([])

        for k, exp, c, s, act, se in self.cfgs:
            output_channel = int(c * width_mult)
            self.blocks.append(InvertedResidual(input_channel, output_channel, k, s, exp, act, se))
            input_channel = output_channel

        self.conv2 = conv_1x1_bn(input_channel, 1280, activation='hardswish')
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, feature_dim)

        self._initialize_weights()

    def forward(self, x, is_feat=False, preact=False):
        out = self.conv1(x)
        f0 = out
        for i, block in enumerate(self.blocks):
            out = block(out)
            if i == 2:
                f1 = out
            if i == 4:
                f2 = out
        out = self.conv2(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        f3 = out
        out = self.classifier(out)

        if is_feat:
            return [f0, f1, f2, f3], out
        else:
            return out

    def get_feat_modules(self):
        feat_m = nn.ModuleList()
        feat_m.append(self.conv1)
        feat_m.extend(self.blocks)
        feat_m.append(self.conv2)
        return feat_m

    def get_bn_before_relu(self):
        # Grab some BNs before activation in the pipeline
        bn1 = self.conv1[1]
        bn2 = self.blocks[2].conv[1]  # block 3
        bn3 = self.blocks[4].conv[1]  # block 5
        bn4 = self.conv2[1]
        return [bn1, bn2, bn3, bn4]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

cfgs_large = [
    [3, 16, 16, 1, 'relu', False],
    [3, 64, 24, 2, 'relu', False],
    [3, 72, 24, 1, 'relu', False],
    [5, 72, 40, 2, 'hardswish', True],
    [5, 120, 40, 1, 'hardswish', True],
    [3, 240, 80, 2, 'hardswish', False],
    [3, 200, 80, 1, 'hardswish', False],
    [3, 184, 80, 1, 'hardswish', False],
    [3, 480, 112, 1, 'hardswish', True],
    [5, 672, 160, 2, 'hardswish', True]
]

cfgs_small = [
    [3, 16, 16, 2, 'relu', False],
    [3, 72, 24, 2, 'relu', False],
    [5, 72, 40, 2, 'hardswish', True],
    [5, 120, 48, 1, 'hardswish', True],
    [5, 240, 96, 2, 'hardswish', True]
]
cfgs_plus = [
        [3, 16, 16, 1, 'relu', False],
        [3, 64, 24, 2, 'relu', False],
        [3, 72, 24, 1, 'relu', False],
        [5, 72, 40, 2, 'hardswish', True],
        [5, 120, 40, 1, 'hardswish', True],
        [3, 240, 80, 2, 'hardswish', True],  # <== SE added
        [3, 200, 80, 1, 'hardswish', True],
        [3, 184, 80, 1, 'hardswish', True],
        [3, 480, 112, 1, 'hardswish', True],
        [5, 672, 160, 2, 'hardswish', True],
        [5, 672, 160, 1, 'hardswish', True],  # <== NEW extra block
    ]

def mobilenetv3_large(num_classes=100):
    return MobileNetV3(cfgs_large, feature_dim=num_classes, width_mult=1.0)

def mobilenetv3_small(num_classes=100):
    return MobileNetV3(cfgs_small, feature_dim=num_classes, width_mult=1.0)

def mobilenetv3_large_plus(num_classes=100, width_mult=1.5):
    return MobileNetV3(cfgs_plus, feature_dim=num_classes, width_mult=width_mult)



if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)
    net = mobilenetv3_large(100)
    feats, logit = net(x, is_feat=True)
    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)
    
    for bn in net.get_bn_before_relu():
        print('BN layer:', isinstance(bn, nn.BatchNorm2d))
