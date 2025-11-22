#!/usr/bin/env python3
"""
train_parking_model_3spots.py

Train a simple CNN for 3 parking spots (front view).

Dataset:
    ./dataset/*.jpg|*.jpeg|*.png

Each image:
    - Shows all 3 spots from the same fixed camera position.
    - Filename starts with 3 digits (0 or 1):

        0 = empty, 1 = full

      Examples:
        000_01.jpg  -> [0,0,0] (all empty)
        111_02.jpg  -> [1,1,1] (all full)
        101_03.jpg  -> [1,0,1] (spot 2 empty)

Output:
    - Trains a model with 3 spots × 2 classes (empty/full) = 6 logits.
    - Saves weights to: parking_model.pt   (same folder as this script)

Run:
    cd Code/Server
    python3 train_parking_model_3spots.py
"""

import os
import glob
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# -------- Model: 3 spots x 2 classes (empty/full) --------

class SimplePatternNetBinary(nn.Module):
    """
    Input : 3x64x64 image (all 3 spots visible)
    Output: 6 logits:
        [s0_empty, s0_full,
         s1_empty, s1_full,
         s2_empty, s2_full]
    """

    def __init__(self, num_spots: int = 3, num_classes: int = 2):
        super().__init__()
        self.num_spots = num_spots
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 64x64 -> two pools -> 16x16
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_spots * num_classes)  # 3 spots * 2 classes = 6 logits

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (B,16,32,32)
        x = self.pool(F.relu(self.conv2(x)))   # (B,32,16,16)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                        # (B, 6)
        return x


# -------- Dataset: filenames like 101_something.jpg --------

class ParkingPatternDataset(Dataset):
    """
    Reads images from ./dataset.
    Uses first `num_spots` chars of filename as labels (0/1).

    Example (num_spots=3):
        000_01.jpg      -> [0,0,0]
        111_full.jpg    -> [1,1,1]
        101_mix_01.png  -> [1,0,1]
    """

    def __init__(self, folder: str = "dataset", num_spots: int = 3, image_size: int = 64):
        self.folder = folder
        self.image_size = image_size
        self.num_spots = num_spots

        paths = []
        paths += glob.glob(os.path.join(folder, "*.jpg"))
        paths += glob.glob(os.path.join(folder, "*.jpeg"))
        paths += glob.glob(os.path.join(folder, "*.png"))

        self.samples = []
        for path in sorted(paths):
            name = os.path.basename(path)

            if len(name) < num_spots:
                print(f"[WARN] Skip {name}: filename too short for {num_spots}-digit label.")
                continue

            code = name[:num_spots]
            if any(c not in "01" for c in code):
                print(f"[WARN] Skip {name}: first {num_spots} chars must be 0/1, got '{code}'.")
                continue

            labels = [int(c) for c in code]  # 0 = empty, 1 = full
            if len(labels) != num_spots:
                continue

            self.samples.append((path, labels))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        print(f"[INFO] Found {len(self.samples)} labeled images in '{folder}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, labels = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)  # (3, H, W)
        y = torch.tensor(labels, dtype=torch.long)  # (num_spots,), each 0 or 1
        return x, y


# -------- Training loop --------

def train(
    data_folder: str = "dataset",
    out_path: str = "parking_model.pt",
    epochs: int = 40,
    batch_size: int = 4,
    lr: float = 1e-3,
    num_spots: int = 3,
):
    device = torch.device("cpu")

    dataset = ParkingPatternDataset(data_folder, num_spots=num_spots)
    if len(dataset) == 0:
        print(f"[ERROR] No valid training images found in '{data_folder}'.")
        print(f"Make sure filenames start with {num_spots} digits of 0/1, e.g. 101_img1.jpg")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimplePatternNetBinary(num_spots=num_spots).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for x, y in loader:
            x = x.to(device)                 # (B,3,64,64)
            y = y.to(device)                 # (B,num_spots) each in {0,1}

            logits = model(x)                # (B,num_spots*2)
            logits = logits.view(-1, num_spots, 2)  # (B, spots, classes)

            # loss: average CE over spots
            loss = 0.0
            for spot in range(num_spots):
                loss += criterion(logits[:, spot, :], y[:, spot])
            loss = loss / float(num_spots)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

        avg_loss = total_loss / max(total_samples, 1)
        print(f"[Epoch {epoch}/{epochs}] loss={avg_loss:.4f}")

    torch.save(model.state_dict(), out_path)
    print(f"[OK] Saved model to {out_path}")


if __name__ == "__main__":
    train()
