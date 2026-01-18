import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast

# --------------------------------------------------
# Dummy dataset: 10000 samples, 3x224x224 images
# --------------------------------------------------
X = torch.randn(10000, 3, 224, 224)
y = torch.randint(0, 10, (10000,))
dataset = TensorDataset(X, y)
loader = DataLoader(
    dataset, batch_size=64, shuffle=True,
    num_workers=4, pin_memory=True, prefetch_factor=2
)

# --------------------------------------------------
# Simple CNN
# --------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# --------------------------------------------------
# Training loop with:
#   • Gradient accumulation
#   • Mixed precision
#   • Automatic memory clearing
# --------------------------------------------------
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()            # handles loss scaling for FP16

accum_steps = 4          # effective batch = 64 * 4 = 256
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()            # start with clean gradients
    for i, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # FP16 autocast
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels) / accum_steps

        # Backward + accumulate
        scaler.scale(loss).backward()

        # Every 'accum_steps' batches, update weights
        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Memory debugging snippet (run once per epoch)
        if i == 0:
            print(torch.cuda.memory_summary(device=device, abbreviated=True))

    # Manual cache clearing – useful for very large models
    torch.cuda.empty_cache()
