import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from transformers import MobileNetV1ForImageClassification

# Parameters
batch_size, input_size, epochs, ft_epochs = 64, 192, 50, 10
fine_tune = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Augmentation (Fixed: RandomErasing now after ToTensor)
transform = {
    "train": transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomCrop(input_size, padding=4),
        transforms.ToTensor(),  # Convert to tensor before applying RandomErasing
        transforms.RandomErasing(p=0.2),  # Now applied correctly
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load CIFAR-100 dataset
data = {
    "train": DataLoader(torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=transform["train"]), batch_size=batch_size, shuffle=True, num_workers=4),
    "test": DataLoader(torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=transform["test"]), batch_size=batch_size, shuffle=False, num_workers=4)
}

# Load pre-trained MobileNetV1 model
model = MobileNetV1ForImageClassification.from_pretrained("google/mobilenet_v1_0.75_192")
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, 100)  # Adjust classifier for CIFAR-100

# Initialize classifier weights using Xavier initialization
nn.init.xavier_uniform_(model.classifier.weight)

# Load pre-trained weights and freeze most of the model
checkpoint = torch.load("mobilenet_v1_0.75_192_model.pth", weights_only=True)
model.load_state_dict({k: v for k, v in checkpoint.items() if "classifier" not in k}, strict=False)
model.to(device)

# Unfreeze last two feature blocks + classifier
for name, param in model.named_parameters():
    if "features.16" in name or "features.15" in name:  # Unfreezing last two feature blocks
        param.requires_grad = True
    else:
        param.requires_grad = False
for param in model.classifier.parameters(): 
    param.requires_grad = True

# Loss function with label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Optimizer and LR scheduler
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Mixup function
def mixup_data(x, y, alpha=0.2):
    """Applies Mixup to the batch."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Training function
def train_model(epochs, fine_tune=False):
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for images, labels in data["train"]:
            images, labels_a, labels_b, lam = mixup_data(images, labels)  # Apply mixup
            images, labels_a, labels_b = images.to(device), labels_a.to(device), labels_b.to(device)

            optimizer.zero_grad()
            outputs = model(images).logits
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)  # Use mixup loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        scheduler.step()  # Adjust learning rate
        losses.append(epoch_loss / len(data["train"]))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {losses[-1]:.4f}")
    
    return losses

# Train classifier
train_losses = train_model(epochs)
torch.save(model.state_dict(), "mobilenet_v1_0.75_192_cifar_top.pth")

# Fine-tuning phase
if fine_tune:
    print("Fine-tuning model...")
    
    # Unfreeze entire model for fine-tuning
    for param in model.parameters(): param.requires_grad = True
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Decay faster

    val_losses = train_model(ft_epochs, fine_tune=True)
    torch.save(model.state_dict(), "mobilenet_v1_0.75_192_cifar_distilled.pth")

# Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in data["test"]:
        images, labels = images.to(device), labels.to(device)
        _, predicted = torch.max(model(images).logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Plot training curves
plt.plot(train_losses, label="Training Loss")
if fine_tune: plt.plot(val_losses, label="Fine-tune Loss")
plt.xlabel("Epochs"), plt.ylabel("Loss"), plt.legend(), plt.title("Training Curve")
plt.savefig("training_curve.png")
