'''
1. Common Setup

# Install once
pip install torch torchvision torchtext matplotlib seaborn tqdm

# Imports that are common to all pipelines
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

#Device selection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

'''
Why?
CUDA accelerates all matrix operations; CPU is fine for small experiments.
By moving data and the model to device you keep all tensors on the same backend.

2. MNIST (handwritten digits)
2.1 Load & inspect
# 1. Download and transform into tensors
'''
mnist_train = datasets.MNIST(root='.', train=True,
                            transform=transforms.ToTensor(),
                            download=True)
mnist_test  = datasets.MNIST(root='.', train=False,
                            transform=transforms.ToTensor(),
                            download=True)

# 2. Basic statistics
print(f"Train: {len(mnist_train)} samples")
print(f"Test : {len(mnist_test)} samples")

# 3. Visualise 5 random images
fig, ax = plt.subplots(1, 5, figsize=(10,2))
for i, ax_i in enumerate(ax):
    idx = np.random.randint(0, len(mnist_train))
    img, label = mnist_train[idx]
    ax_i.imshow(img.squeeze(), cmap='gray')
    ax_i.set_title(f"label={label}")
    ax_i.axis('off')
plt.show()
'''
Why?

datasets.MNIST automatically downloads the dataset if not present.
transforms.ToTensor() converts PIL images into C×H×W tensors with values in [0,1].
Inspecting data helps you see class distribution, image quality, and spot outliers.
2.2 DataLoaders
'''
batch_size = 64
train_loader = DataLoader(mnist_train, batch_size=batch_size,
                          shuffle=True, num_workers=2)
test_loader  = DataLoader(mnist_test,  batch_size=batch_size,
                          shuffle=False, num_workers=2)
'''
Why?
Shuffling improves generalisation by breaking order.
num_workers speeds up data loading on multi‑core machines.

2.3 Model – Small CNN
'''
class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        # Convolution 1: 1 input channel -> 32 output channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Convolution 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)      # 10 output classes

    def forward(self, x):
        # Convolution → ReLU → Pool
        x = self.pool(F.relu(self.conv1(x)))   # size: B×32×14×14
        x = self.pool(F.relu(self.conv2(x)))   # size: B×64×7×7
        x = x.view(-1, 64 * 7 * 7)             # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                        # logits
        return x
'''
Why the architecture?

Conv layers learn local pixel patterns.
Pooling reduces spatial resolution, making the network more robust to small shifts.
The network is tiny enough that training takes only a few seconds per epoch, yet it reaches ~99 % accuracy.
view(-1, 64*7*7) flattens the feature map into a vector for the fully‑connected part.
2.4 Instantiate, criterion, optimizer
'''
model = MnistCNN().to(device)

criterion = nn.CrossEntropyLoss()   # combines LogSoftmax + NLL
optimizer = optim.Adam(model.parameters(), lr=0.001)
'''
Why?
CrossEntropyLoss is the standard for multi‑class classification.
Adam adapts the learning rate per‑parameter, which speeds convergence on small data‑sets.

# 2.5 Training loop
'''
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0
    for inputs, targets in tqdm(loader, desc='train'):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()              # 1️⃣ Zero gradients
        outputs = model(inputs)            # 2️⃣ Forward
        loss = criterion(outputs, targets) # 3️⃣ Loss
        loss.backward()                    # 4️⃣ Backward
        optimizer.step()                   # 5️⃣ Update weights

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == targets.data)
        total   += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct.double() / total
    return epoch_loss, epoch_acc.item()
'''
Why each step?

Zero gradients – otherwise they accumulate across batches.
Forward pass gives logits.
Loss compares logits to true labels.
Backward computes ∂L/∂θ.
Optimizer updates θ using Adam’s rule.
#2.6 Validation
'''
def evaluate(model, loader, device):
    model.eval()
    loss_total = 0.0
    correct    = 0
    total      = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_total += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == targets.data)
            total   += inputs.size(0)
    return loss_total / total, correct.double() / total
#2.7 Full training script
num_epochs = 10
for epoch in range(num_epochs):
    tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
          f"val   loss={val_loss:.4f} acc={val_acc:.4f}")
'''
What to expect?
After ~10 epochs you should reach ~99 % test accuracy.
Loss should drop below 0.1 quickly.

2.8 Iteration – What to tweak?
What	How to tweak	What you’ll see
Learning rate	Reduce to 1e‑4 or add a scheduler (StepLR).	Faster convergence but risk of plateaus.
Batch size	Increase to 128 or 256 for GPU; decrease to 32 for noisy data.	Larger batches smooth gradients, but may need more epochs.
Dropout	Add nn.Dropout(p=0.25) after fc1.	Helps if you overfit.
Data augmentation	Random rotations, shifts, etc.	Makes the model robust to slight writing variations.
Early stopping	Track validation loss; stop if no improvement for 3 epochs.	Avoids over‑training.
'''