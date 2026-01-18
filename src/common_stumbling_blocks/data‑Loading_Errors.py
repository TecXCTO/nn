import os
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class RobustImageFolder(Dataset):
    """
    A drop‑in replacement for torchvision.datasets.ImageFolder
    that skips corrupted files, handles missing files,
    and logs issues.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir).resolve()
        self.transform = transform

        # Build list of (image_path, label) tuples
        self.samples = []
        for class_idx, class_name in enumerate(sorted(os.listdir(self.root_dir))):
            class_path = self.root_dir / class_name
            if not class_path.is_dir():
                continue
            for img_name in os.listdir(class_path):
                img_path = class_path / img_name
                if img_path.is_file():
                    self.samples.append((img_path, class_idx))

        if not self.samples:
            raise RuntimeError(f"No images found in {self.root_dir}")

        self.n_samples = len(self.samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")  # enforce 3‑channel
        except Exception as e:
            # Log and replace corrupted image with a random image from dataset
            print(f"[WARNING] Failed to load {img_path}: {e}")
            # Pick a random index that isn't the same
            alt_idx = (idx + 1) % self.n_samples
            img_path, label = self.samples[alt_idx]
            with Image.open(img_path) as img:
                img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

# ---------------------------------------
# Usage
# ---------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = RobustImageFolder(root_dir="data/train", transform=transform)

# DataLoader with deterministic workers (good for debugging)
def worker_init_fn(worker_id):
    # Set the same seed for each worker to make errors reproducible
    seed = torch.initial_seed() + worker_id
    random.seed(seed)
    torch.manual_seed(seed)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    worker_init_fn=worker_init_fn
)
