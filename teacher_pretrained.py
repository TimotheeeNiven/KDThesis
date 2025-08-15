import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import vit_pretrained  # Make sure this is in __init__.py and model_dict
import os

# === Device Setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Load Pretrained Hugging Face ViT Model ===
model = vit_pretrained(num_classes=100)
model.eval().to(device)

# === CIFAR-100 Validation Loader ===
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

val_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=1)

# === Accuracy Function ===
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    return [correct[:, :k].float().sum().item() * 100.0 / target.size(0) for k in topk]

# === Run Evaluation ===
top1, top5, total = 0.0, 0.0, 0
with torch.no_grad():
    for idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)  # ? No longer using `.logits`
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        top1 += acc1 * inputs.size(0)
        top5 += acc5 * inputs.size(0)
        total += inputs.size(0)
        print(f"Batch {idx + 1:03d}: Top-1 = {acc1:.2f}%, Top-5 = {acc5:.2f}%")

# === Final Output ===
print(f"\n? ViT-Base on CIFAR-100:")
print(f"   Top-1 Accuracy: {top1 / total:.2f}%")
print(f"   Top-5 Accuracy: {top5 / total:.2f}%")

# === Save Model Checkpoint ===
save_path = '/users/rniven1/GitHubRepos/RepDistiller/models/vit_pretrained_cifar100_lr_0.05_decay_0.0005_trial_0'
os.makedirs(save_path, exist_ok=True)
torch.save({
    'model': model.state_dict(),
    'arch': 'vit_pretrained',
    'num_classes': 100,
    'source': 'pkr7098/cifar100-vit-base-patch16-224-in21k'
}, os.path.join(save_path, 'vit_pretrained.pth'))

print(f"\n? Pretrained model saved to: {save_path}/vit_teacher.pth")
