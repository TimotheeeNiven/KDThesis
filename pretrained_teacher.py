import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import convnext_small  # your wrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Load model ===
model = convnext_small(pretrained=True, num_classes=100)
model.eval().to(device)

# === CIFAR-100 Loader ===
transform = transforms.Compose([
    transforms.Resize(224),  # resize for ConvNeXt input
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

val_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4)

# === Accuracy Function ===
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    return [correct[:, :k].float().sum().item() * 100.0 / target.size(0) for k in topk]

# === Run Validation ===
top1, top5, total = 0, 0, 0
with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        top1 += acc1 * inputs.size(0)
        top5 += acc5 * inputs.size(0)
        total += inputs.size(0)

print(f"\n? CIFAR-100 Accuracy: Top-1 = {top1/total:.2f}%, Top-5 = {top5/total:.2f}%")
