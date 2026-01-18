import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import os

# --------------------------
# Basic dataset & transforms
# --------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
train_dataset = datasets.CIFAR10(
    root='data', train=True, download=True, transform=train_transform
)
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=2
)

# --------------------------
# Define model (simple CNN)
# --------------------------
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# --------------------------
# Objective function for Optuna
# --------------------------
def objective(trial):
    # Suggest hyper‑parameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    momentum = trial.suggest_uniform('momentum', 0.5, 0.99)

    # Re‑create dataloader with suggested batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    # Build model, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # Simple training loop (1 epoch per trial)
    model.train()
    epoch_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Early stopping / pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    avg_loss = epoch_loss / len(train_loader)

    # For demonstration, we use validation loss as the objective
    # In practice, compute val_loss on a hold‑out set
    return avg_loss

# --------------------------
# Running the study
# --------------------------
if __name__ == "__main__":
    storage_name = "sqlite:///optuna_study.db"
    study = optuna.create_study(
        study_name="cifar10_optuna",
        storage=storage_name,
        direction="minimize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=30, timeout=3600)  # 1 hour max

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
