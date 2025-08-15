from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .resnetv2 import ResNet50, ResNet101, ResNet152
from .resnext import resnext50_32x4d, resnext101_32x8d
from .regnet import regnety_4gf, regnety_16gf, regnety_32gf
from .wrn import (
    wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2, wrn_40_4, wrn_28_10, wrn_28_1, wrn_28_4,
    wrn_20_4, wrn_40_10, wrn_10_2, wide_resnet101_2
)
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .mobilenetv1 import mobilev1_half
from .mobilenetv2 import mobile_half
from .mobilenetv3 import mobilenetv3_large, mobilenetv3_large_plus
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
from .efficientnet import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,
    efficientnet_b0_untrained, efficientnet_b1_untrained, efficientnet_b2_untrained,
    efficientnet_b3_untrained, efficientnet_b4_untrained, efficientnet_b5_untrained,
    efficientnet_b6_untrained, efficientnet_b7_untrained
)
from .efficientnet_v2_m import efficientnet_v2_m
from .vit import (
    vit_tiny, vit_small, vit_base, vit_medium, vit_distilled,
    vit_pretrained, vit_pretrained_imagetiny
)
from .convnext import convnext_tiny, convnext_large, convnext_base, convnext_small
from .convnext_custom import convnext_tiny_for_large, convnext_tiny_for_base, convnext_tiny_for_small
from .densenet import densenet121, densenet169
from .pyramidnet import pyramidnet272, pyramidnet110
from .bert import roberta, distilbert, tinybert, mobilebert, albert, bertbase, bertsmall

import torch
import torch.nn as nn

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'ResNet152': ResNet152,
    'regnety16gf': regnety_16gf,
    'regnety4gf': regnety_4gf,
    'regnety32gf': regnety_32gf,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'wrn_40_4': wrn_40_4,
    'wrn_28_10': wrn_28_10,
    'wrn_28_1': wrn_28_1,
    'wrn_28_4': wrn_28_4,
    'wrn_20_4': wrn_20_4,
    'wrn_40_10': wrn_40_10,
    'wrn_10_2': wrn_10_2,
    'wide': wide_resnet101_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV1': mobilev1_half,
    'MobileNetV2': mobile_half,
    'MobileNetV3': mobilenetv3_large,
    'MobileNetV3.1': mobilenetv3_large_plus,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'efficientnet_b0': efficientnet_b0,
    'efficientnet_b1': efficientnet_b1,
    'efficientnet_b2': efficientnet_b2,
    'efficientnet_b3': efficientnet_b3,
    'efficientnet_b4': efficientnet_b4,
    'efficientnet_b5': efficientnet_b5,
    'efficientnet_b6': efficientnet_b6,
    'efficientnet_b7': efficientnet_b7,
    'efficientnet_b0_untrained': efficientnet_b0_untrained,
    'efficientnet_b1_untrained': efficientnet_b1_untrained,
    'efficientnet_b2_untrained': efficientnet_b2_untrained,
    'efficientnet_b3_untrained': efficientnet_b3_untrained,
    'efficientnet_b4_untrained': efficientnet_b4_untrained,
    'efficientnet_b5_untrained': efficientnet_b5_untrained,
    'efficientnet_b6_untrained': efficientnet_b6_untrained,
    'efficientnet_b7_untrained': efficientnet_b7_untrained,
    'efficientnet_v2m': efficientnet_v2_m,
    'convnexttiny4large': convnext_tiny_for_large,
    'convnexttiny4small': convnext_tiny_for_small,
    'convnexttiny4base': convnext_tiny_for_base,
    'convnexttiny': convnext_tiny,
    'convnextlarge': convnext_large,
    'convnextbase': convnext_base,
    'convnextsmall': convnext_small,
    'densenet121': densenet121,
    'densenet169': densenet169,
    'pyramidnet272': pyramidnet272,
    'pyramidnet110': pyramidnet110,
    'roberta': roberta,
    'distilbert': distilbert,
    'tinybert': tinybert,
    'mobilebert': mobilebert,
    'albert': albert,
    'bertbase': bertbase,
    'bertsmall': bertsmall,
    'vit_tiny': vit_tiny,
    'vit_small': vit_small,
    'vit_base': vit_base,
    'vit_medium': vit_medium,
    'vit_distilled': vit_distilled,
    'vit_pretrained': vit_pretrained,
    'vit_pretrainedimagetiny': vit_pretrained_imagetiny,
}
