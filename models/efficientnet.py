import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, AutoConfig

__all__ = [
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
    'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
    'efficientnet_b6', 'efficientnet_b7',
    'efficientnet_b0_untrained', 'efficientnet_b1_untrained', 'efficientnet_b2_untrained',
    'efficientnet_b3_untrained', 'efficientnet_b4_untrained', 'efficientnet_b5_untrained',
    'efficientnet_b6_untrained', 'efficientnet_b7_untrained',
]


class EfficientNetWrapper(nn.Module):
    def __init__(self, model_name="google/efficientnet-b0", num_classes=1000, pretrained=True):
        super(EfficientNetWrapper, self).__init__()

        if pretrained:
            self.model = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
            self.model = AutoModelForImageClassification.from_config(config)

    def forward(self, x):
        outputs = self.model(x)
        return outputs.logits


# Pretrained versions
def efficientnet_b0(num_classes=1000): return EfficientNetWrapper("google/efficientnet-b0", num_classes=num_classes, pretrained=True)
def efficientnet_b1(num_classes=1000): return EfficientNetWrapper("google/efficientnet-b1", num_classes=num_classes, pretrained=True)
def efficientnet_b2(num_classes=1000): return EfficientNetWrapper("google/efficientnet-b2", num_classes=num_classes, pretrained=True)
def efficientnet_b3(num_classes=1000): return EfficientNetWrapper("google/efficientnet-b3", num_classes=num_classes, pretrained=True)
def efficientnet_b4(num_classes=1000): return EfficientNetWrapper("google/efficientnet-b4", num_classes=num_classes, pretrained=True)
def efficientnet_b5(num_classes=1000): return EfficientNetWrapper("google/efficientnet-b5", num_classes=num_classes, pretrained=True)
def efficientnet_b6(num_classes=1000): return EfficientNetWrapper("google/efficientnet-b6", num_classes=num_classes, pretrained=True)
def efficientnet_b7(num_classes=1000): return EfficientNetWrapper("google/efficientnet-b7", num_classes=num_classes, pretrained=True)

# Untrained versions
def efficientnet_b0_untrained(num_classes=1000): return EfficientNetWrapper("google/efficientnet-b0", num_classes=num_classes, pretrained=False)
def efficientnet_b1_untrained(num_classes=1000): return EfficientNetWrapper("google/efficientnet-b1", num_classes=num_classes, pretrained=False)
def efficientnet_b2_untrained(num_classes=1000): return EfficientNetWrapper("google/efficientnet-b2", num_classes=num_classes, pretrained=False)
def efficientnet_b3_untrained(num_classes=1000): return EfficientNetWrapper("google/efficientnet-b3", num_classes=num_classes, pretrained=False)
def efficientnet_b4_untrained(num_classes=1000): return EfficientNetWrapper("google/efficientnet-b4", num_classes=num_classes, pretrained=False)
def efficientnet_b5_untrained(num_classes=1000): return EfficientNetWrapper("google/efficientnet-b5", num_classes=num_classes, pretrained=False)
def efficientnet_b6_untrained(num_classes=1000): return EfficientNetWrapper("google/efficientnet-b6", num_classes=num_classes, pretrained=False)
def efficientnet_b7_untrained(num_classes=1000): return EfficientNetWrapper("google/efficientnet-b7", num_classes=num_classes, pretrained=False)


if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    model = efficientnet_b0()
    logits = model(x)
    print("Pretrained:", logits.shape)

    model_untrained = efficientnet_b0_untrained()
    logits_untrained = model_untrained(x)
    print("Untrained:", logits_untrained.shape)
