import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from timm import create_model

# Parameters
batch_size = 64
input_size = 224  # WideResNet101_2 requires 224x224 input
epochs = 20
ft_epochs = 10
fine_tune = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
transform_train = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # Added augmentation for robustness
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR-100 dataset
train_dataset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load WideResNet101_2 model
print("Loading pre-trained WideResNet101_2 model...")
model = create_model("wide_resnet101_2.tv_in1k", pretrained=True)
in_features = model.get_classifier().in_features
model.fc = nn.Linear(in_features, 100)  # Adjust for CIFAR-100
model.to(device)

# Freeze base model parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only the classifier layer
for param in model.fc.parameters():
    param.requires_grad = True

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

# Function to evaluate model accuracy
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Train the top layers
print("Training the top layer...")
model.train()
train_losses = []

for epoch in range(epochs):
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    test_accuracy = evaluate_model(model, test_loader)
    train_losses.append(train_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

torch.save(model.state_dict(), "wide_resnet101_2_cifar_top.pth")

# Fine-tuning (unfreeze base model)
if fine_tune:
    print("Fine-tuning the entire model...")
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    val_losses = []
    
    for epoch in range(ft_epochs):
        fine_tune_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            fine_tune_loss += loss.item()
        
        avg_loss = fine_tune_loss / len(train_loader)
        val_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        test_accuracy = evaluate_model(model, test_loader)
        print(f"Fine-tune Epoch {epoch + 1}/{ft_epochs}, Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    torch.save(model.state_dict(), "wide_resnet101_2_cifar_distilled.pth")
