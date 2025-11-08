#!/usr/bin/env python3
"""
train_parking_model.py

Train a simple CNN for 4 parking spots (front view).

Dataset:
    ./dataset/*.jpg|*.jpeg|*.png

Each image:
    - Shows all 4 spots from the same fixed camera position.
    - Filename starts with 4 digits (0 or 1):

        0 = empty, 1 = full

      Example:
        0000_01.jpg  -> [0,0,0,0]
        1111_02.jpg  -> [1,1,1,1]
        0101_test.jpg-> [0,1,0,1]

Output:
    - Trains a model with 4 spots × 2 classes (empty/full) = 8 logits.
    - Saves weights to: parking_model.pt   (same folder as this script)

Run:
    cd Code/Server
    python3 train_parking_model.py
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


# -------- Model: 4 spots x 2 classes (empty/full) --------

class SimplePatternNetBinary(nn.Module):
    """
    Input : 3x64x64 image (all 4 spots visible)
    Output: 8 logits:
        [s0_empty, s0_full,
         s1_empty, s1_full,
         s2_empty, s2_full,
         s3_empty, s3_full]
    """

    def __init__(self, num_spots: int = 4, num_classes: int = 2):
        super().__init__()
        self.num_spots = num_spots
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 64x64 -> two pools -> 16x16
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_spots * num_classes)  # 8 logits

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (B,16,32,32)
        x = self.pool(F.relu(self.conv2(x)))   # (B,32,16,16)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                        # (B, 8)
        return x


# -------- Dataset: filenames like 0101_something.jpg --------

class ParkingPatternDataset(Dataset):
    """
    Reads images from ./dataset.
    Uses first 4 chars of filename as labels (0/1).

    Example:
        0000_01.jpg      -> [0,0,0,0]
        1111_full.jpg    -> [1,1,1,1]
        0101_mix_01.png  -> [0,1,0,1]
    """

    def __init__(self, folder: str = "dataset", image_size: int = 64):
        self.folder = folder
        self.image_size = image_size

        paths = []
        paths += glob.glob(os.path.join(folder, "*.jpg"))
        paths += glob.glob(os.path.join(folder, "*.jpeg"))
        paths += glob.glob(os.path.join(folder, "*.png"))

        self.samples = []
        for path in sorted(paths):
            name = os.path.basename(path)

            if len(name) < 4:
                print(f"[WARN] Skip {name}: filename too short for 4-digit label.")
                continue

            code = name[:4]
            if any(c not in "01" for c in code):
                print(f"[WARN] Skip {name}: first 4 chars must be 0/1, got '{code}'.")
                continue

            labels = [int(c) for c in code]  # 0 = empty, 1 = full
            if len(labels) != 4:
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
        y = torch.tensor(labels, dtype=torch.long)  # (4,), each 0 or 1
        return x, y


# -------- Training loop --------

def train(
    data_folder: str = "dataset",
    out_path: str = "parking_model.pt",
    epochs: int = 40,
    batch_size: int = 4,
    lr: float = 1e-3,
):
    device = torch.device("cpu")

    dataset = ParkingPatternDataset(data_folder)
    if len(dataset) == 0:
        print(f"[ERROR] No valid training images found in '{data_folder}'.")
        print("Make sure filenames start with 4 digits of 0/1, e.g. 0101_img1.jpg")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimplePatternNetBinary().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for x, y in loader:
            x = x.to(device)          # (B,3,64,64)
            y = y.to(device)          # (B,4) each in {0,1}

            logits = model(x)         # (B,8)
            logits = logits.view(-1, 4, 2)  # (B, spots, classes)

            # loss: average CE over 4 spots
            loss = 0.0
            for spot in range(4):
                loss += criterion(logits[:, spot, :], y[:, spot])
            loss = loss / 4.0

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
