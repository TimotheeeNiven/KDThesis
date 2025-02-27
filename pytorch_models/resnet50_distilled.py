import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.models import resnet50

# Parameters
batch_size, input_size, epochs, ft_epochs = 64, 224, 20, 10
fine_tune = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
transform = {
    "train": transforms.Compose([
        transforms.Resize((input_size, input_size)),  # Resize CIFAR-100 (32x32 -> 224x224)
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomCrop(input_size, padding=4),
        transforms.ToTensor(),
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
    "train": DataLoader(
        torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=transform["train"]),
        batch_size=batch_size, shuffle=True, num_workers=4
    ),
    "test": DataLoader(
        torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=transform["test"]),
        batch_size=batch_size, shuffle=False, num_workers=4
    )
}

# Load ResNet-50 model (trained on ImageNet)
model = resnet50()
checkpoint = torch.load("resnet50_image.pth", map_location=device)

# Load pre-trained ImageNet weights (excluding classifier)
model.load_state_dict({k: v for k, v in checkpoint.items() if "fc" not in k}, strict=False)

# Adjust classifier for CIFAR-100
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 100)  # 100 classes for CIFAR-100
model.to(device)

# Freeze all layers except the classifier
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

def train_model(epochs, fine_tune=False):
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for images, labels in data["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(data["train"]))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {losses[-1]:.4f}")
    return losses

# Train classifier
train_losses = train_model(epochs)
torch.save(model.state_dict(), "resnet50_cifar100_top.pth")

# Fine-tuning
if fine_tune:
    print("Fine-tuning model...")
    for param in model.parameters():
        param.requires_grad = True  # Unfreeze entire model
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    val_losses = train_model(ft_epochs, fine_tune=True)
    torch.save(model.state_dict(), "resnet50_cifar100_finetuned.pth")

# Evaluate
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in data["test"]:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Plot training curves
plt.plot(train_losses, label="Training Loss")
if fine_tune: plt.plot(val_losses, label="Fine-tune Loss")
plt.xlabel("Epochs"), plt.ylabel("Loss"), plt.legend(), plt.title("Training Curve")
plt.savefig("training_curve.png")
