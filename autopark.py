#!/usr/bin/env python3
"""
autopark.py

Design:

- Exactly 4 fixed parking spots visible in front of the car.
- One trained deep learning model:
    Input: single front image (all 4 spots visible).
    Output: 4 x 2 logits  -> for each spot: [empty, full].
- Runtime behavior:
    1) Drive forward until ultrasonic distance <= APPROACH_DISTANCE.
    2) Stop & capture ONE front image.
    3) Use model to classify each of the 4 spots as "empty" or "full".
    4) If all 4 are "full": print "All spots are full." and exit.
    5) If one or more are "empty": choose one (first empty) and
       execute a slow parking maneuver into that spot,
       using ultrasonic to avoid collisions.

This is a scaffold:
- You train parking_model.pt separately.
- Movement timings MUST be tuned on your hardware.
"""

import os
import sys
import time
import io
from typing import List, Optional

from motor import Ordinary_Car

# ---------- Ultrasonic import (support both filenames) ----------
try:
    from ultrasonic import Ultrasonic
except ImportError:
    try:
        from Ultrasonic import Ultrasonic
    except ImportError:
        Ultrasonic = None

# ---------- Camera ----------
try:
    from camera import Camera
except ImportError:
    Camera = None

# ---------- PIL for image handling ----------
try:
    from PIL import Image
except ImportError:
    Image = None

# ---------- PyTorch for model ----------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
except ImportError:
    torch = None
    nn = None
    F = None
    transforms = None


# ============================================================
# 4 x 2 logits model: 4 spots x {empty, full}
# ============================================================

class SimplePatternNetBinary(nn.Module if nn is not None else object):
    """
    Example CNN:
    - Input: 3x64x64 full-frame image (all 4 spots visible)
    - Output: 8 logits = 4 spots * 2 classes (empty, full)
      Layout: [s0_empty, s0_full, s1_empty, s1_full, s2_empty, s2_full, s3_empty, s3_full]
    """

    def __init__(self, num_spots: int = 4, num_classes: int = 2):
        if nn is None:
            return
        super().__init__()
        self.num_spots = num_spots
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_spots * num_classes)  # 8 logits

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (B,16,32,32)
        x = self.pool(F.relu(self.conv2(x)))   # (B,32,16,16)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                        # (B, 8)
        return x


class ParkingPatternModelBinary:
    """
    Wraps the trained 4x2 model.

    - Expects 'parking_model.pt' in same folder.
    - Outputs 4 labels: "empty" or "full".
    - Each spot is classified independently:
        logits -> reshape (4,2) -> softmax per spot.
    """

    LABELS = ["empty", "full"]  # index 0 = empty, 1 = full

    def __init__(self, model_path: str = "parking_model.pt"):
        self.model_path = model_path

        self.enabled = (
            torch is not None
            and nn is not None
            and transforms is not None
            and os.path.exists(self.model_path)
        )

        if not self.enabled:
            print("[ParkingModel] Deep learning disabled (no torch/model). "
                  "All spots will be treated as full.")
            self.model = None
            return

        self.device = torch.device("cpu")
        self.num_spots = 4
        self.num_classes = 2

        self.model = SimplePatternNetBinary(
            num_spots=self.num_spots,
            num_classes=self.num_classes
        ).to(self.device)

        try:
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            print("[ParkingModel] Loaded model from", self.model_path)
        except Exception as e:
            print("[ParkingModel] Failed to load model:", e)
            self.model = None
            self.enabled = False

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def predict_all(self, image: Optional[Image.Image]) -> List[str]:
        """
        Return list of 4 strings: each "empty" or "full".
        If disabled or no image -> all "full" (safe).
        """
        if not self.enabled or self.model is None or image is None:
            return ["full"] * 4

        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)           # shape: (1, 8)
            logits = logits.view(4, 2)       # (spots, classes)
            probs = torch.softmax(logits, dim=1)

        labels = []
        for i in range(4):
            idx = int(torch.argmax(probs[i]).item())  # 0 or 1
            labels.append(self.LABELS[idx])

        print("[ParkingModel] Spot labels:", labels)
        return labels


# ============================================================
# Auto Park Controller
# ============================================================

class AutoParkController:
    def __init__(self):
        self.car = Ordinary_Car()

        # Ultrasonic
        self.ultra = Ultrasonic() if Ultrasonic is not None else None
        if self.ultra is None:
            print("[AutoPark] WARNING: Ultrasonic missing. Autopark unsafe.")

        # Camera
        self.camera = Camera() if Camera is not None else None
        if self.camera is None:
            print("[AutoPark] WARNING: Camera missing. Autopark disabled.")

        # Model
        self.model = ParkingPatternModelBinary(
