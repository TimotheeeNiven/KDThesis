from transformers import AutoImageProcessor, ResNetForImageClassification
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

# Load CIFAR-100 dataset
dataset = load_dataset("cifar100", split="train[:80%]")
val_dataset = load_dataset("cifar100", split="train[80%:]")
test_dataset = load_dataset("cifar100", split="test")

# Preprocessing
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing for ResNet
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),  # Normalize
])

def preprocess_dataset(dataset, transform):
    def transform_fn(example):
        # Convert the list to an actual image if needed
        img = Image.fromarray(example["img"]) if isinstance(example["img"], list) else example["img"]
        example["pixel_values"] = transform(img)
        return example

    return dataset.with_transform(transform_fn)

train_dataset = preprocess_dataset(dataset, image_transform)
val_dataset = preprocess_dataset(val_dataset, image_transform)
test_dataset = preprocess_dataset(test_dataset, image_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load pre-trained ResNet model
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
print("Original model classifier:", model.classifier)

# Modify the classification head for 100 classes
# Access the Linear layer in the Sequential module
num_features = model.classifier[1].in_features  # Access the in_features of the Linear layer
model.classifier[1] = nn.Linear(num_features, 100)  # Replace it with a new Linear layer for 100 classes

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        inputs = batch["pixel_values"].to(device)
        labels = batch["fine_label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Save the fine-tuned model
torch.save(model.state_dict(), "resnet_cifar100.pth")
