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

'''


# 3. CIFAR‑10 (color images)


# 3.1 Load & transform

cifar_train = datasets.CIFAR10(root='.', train=True,
                               transform=transforms.ToTensor(),
                               download=True)
cifar_test  = datasets.CIFAR10(root='.', train=False,
                               transform=transforms.ToTensor(),
                               download=True)

print(f"Train: {len(cifar_train)} samples")
print(f"Test : {len(cifar_test)} samples")

# 3.2 Inspect a batch
# Show a grid of 8×8 samples

fig, ax = plt.subplots(8, 8, figsize=(10,10))
for i, ax_i in enumerate(ax.flat):
    idx = np.random.randint(0, len(cifar_train))
    img, label = cifar_train[idx]
    ax_i.imshow(np.moveaxis(img.numpy(), 0, -1))   # move C→last
    ax_i.set_title(f"{cifar_train.classes[label]}")
    ax_i.axis('off')
plt.tight_layout()
plt.show()
'''
Why?
CIFAR‑10 images are small and noisy; visualising helps you gauge the difficulty of the task.

3.3 Pre‑processing & augmentation
'''
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),      # random left/right flip
    transforms.RandomCrop(32, padding=4),   # random crop with padding
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))  # mean/std per channel
])

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

cifar_train = datasets.CIFAR10(root='.', train=True,
                               transform=train_tf, download=True)
cifar_test  = datasets.CIFAR10(root='.', train=False,
                               transform=test_tf, download=True)
'''
Why?
RandomCrop + Flip increase data diversity.
Normalising to zero‑mean, unit‑variance accelerates training because every channel shares a similar scale.

3.4 DataLoaders
'''
batch_size = 128
train_loader = DataLoader(cifar_train, batch_size=batch_size,
                          shuffle=True, num_workers=4)
test_loader  = DataLoader(cifar_test, batch_size=batch_size,
                          shuffle=False, num_workers=4)
'''
3.5 Model – Tiny CNN (or use a pretrained ResNet)
Below is a hand‑crafted small network that still reaches ~72 % test accuracy. Feel free to switch to a pretrained ResNet‑18 later.
'''
class CIFARNet(nn.Module):
    def __init__(self):
        super(CIFARNet, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        # Block 2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128,128,3,padding=1)
        self.pool2 = nn.MaxPool2d(2,2)
        # Block 3
        self.conv5 = nn.Conv2d(128,256,3,padding=1)
        self.conv6 = nn.Conv2d(256,256,3,padding=1)
        self.pool3 = nn.MaxPool2d(2,2)
        # Classifier
        self.fc1   = nn.Linear(256*4*4, 512)
        self.fc2   = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool2(F.relu(self.conv4(x)))
        x = self.pool3(F.relu(self.conv5(x)))
        x = self.pool3(F.relu(self.conv6(x)))
        x = torch.flatten(x, 1)           # (batch, 256*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                   # logits
        return x
'''
Why?

3 × 3 kernels are ubiquitous in CNNs; they capture local edges.
Every block has conv → ReLU → pool; this keeps the spatial resolution halving twice → final feature map 4 × 4.
A large fully‑connected layer (512 units) lets the network combine high‑level features across channels.
3.6 Instantiate, criterion, optimizer
'''
model = CIFARNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3.7 Learning‑rate scheduler

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

'''
# 3.8 Training & evaluation functions
Same as before (train_epoch, evaluate). Add torch.no_grad() for validation.

#3.9 Training script
'''
num_epochs = 60
for epoch in range(num_epochs):
    tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, test_loader, device)
    scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
          f"val   loss={val_loss:.4f} acc={val_acc:.4f}")

'''
What to expect?
After ~60 epochs you should see ~70 %–73 % test accuracy.
If you switch to ResNet‑18 (pre‑trained on ImageNet) you’ll surpass 80 % with a modest fine‑tuning.

3.10 Iteration – Common tweaks
Idea	How to implement	Why it matters
Use a deeper network	Add more conv layers or use ResNet / DenseNet.	Higher capacity → higher accuracy (up to ~85 %).
Dropout	Add nn.Dropout(0.5) before fc1.	Helps regularisation.
Learning‑rate schedule	Use CosineAnnealingLR or ReduceLROnPlateau.	Smooths training, avoids sharp jumps.
Batch‑norm	Add nn.BatchNorm2d after each conv.	Stabilises training, especially with small inputs.
Label smoothing	Replace criterion with LabelSmoothingCrossEntropy.	Reduces over‑confidence, can improve generalisation.
Ensembling	Train several models (different seeds) and average predictions.	Often yields 1–2 % boost.
'''