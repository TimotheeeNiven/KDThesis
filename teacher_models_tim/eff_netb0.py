import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import ProgressBar

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Augmentation (Same as Ignite Notebook)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
])

# Load CIFAR-100 dataset
trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

# Load EfficientNet-B2 (CHANGE from Ignite Notebook)
model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)

# Modify classifier for CIFAR-100 (100 classes instead of 1000)
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.5),  # Matches Ignite finetuning dropout
    nn.Linear(num_ftrs, 100)
)
model = model.to(device)

# Define Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-3)

# Learning Rate Scheduler (Same as Ignite)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.003, steps_per_epoch=len(trainloader), epochs=150, pct_start=0.3
)

# Ignite Trainer Function
trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

# Add Learning Rate Scheduling
@trainer.on(Events.ITERATION_COMPLETED)
def update_lr(engine):
    scheduler.step()

# Ignite Evaluator for Accuracy & Loss
evaluator = create_supervised_evaluator(model, metrics={
    "accuracy": Accuracy(),
    "loss": Loss(criterion)
}, device=device)

# Add Model Checkpointing (Same as Ignite)
checkpoint_handler = ModelCheckpoint(
    dirname="./checkpoints", filename_prefix="best_model",
    save_interval=1, n_saved=3, create_dir=True,
    score_function=lambda engine: engine.state.metrics["accuracy"],
    score_name="val_accuracy", global_step_transform=lambda *_: trainer.state.epoch
)
evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {"model": model})

# Progress Bar (Same as Ignite)
pbar = ProgressBar()
pbar.attach(trainer, output_transform=lambda x: {"loss": x})

# Early Stopping (Same as Ignite)
early_stopping = EarlyStopping(patience=10, score_function=lambda engine: engine.state.metrics["accuracy"], trainer=trainer)
evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stopping)

# Training Loop (Same as Ignite)
@trainer.on(Events.EPOCH_COMPLETED)
def validate(engine):
    evaluator.run(testloader)
    metrics = evaluator.state.metrics
    print(f"Validation - Epoch {engine.state.epoch}: Accuracy: {metrics['accuracy']:.4f}, Loss: {metrics['loss']:.4f}")

# Run Training (Same as Ignite)
trainer.run(trainloader, max_epochs=150)

# Final Evaluation
evaluator.run(testloader)
final_accuracy = evaluator.state.metrics["accuracy"]
print(f"Final Model Accuracy: {final_accuracy:.4f}")
