#!/usr/bin/env python3
"""
check_parking_spot.py

Test script to verify the trained parking model.

Usage:
  1. Place the car about 60 cm in front of the 3 parking spots.
  2. Run:  python3 check_parking_spot.py
  3. The script:
       - Captures one frame from the camera
       - Runs the CNN model (parking_model.pt)
       - Prints which spots are predicted empty
       - Prints which spot autopilot would choose (lowest empty index)
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# ---- Camera import (same logic as dashboard) ----
try:
    from Camera_1 import Camera
except ImportError:
    try:
        from camera_1 import Camera
    except ImportError:
        Camera = None


# ============== CONFIG ==============

NUM_SPOTS = 3
IMAGE_SIZE = 64
MODEL_FILENAME = "parking_model.pt"


# ============== MODEL ==============

class SimplePatternNetBinary(nn.Module):
    """
    Same architecture as in train_parking_model_3spots.py

    Input : 3x64x64 image (all 3 spots visible)
    Output: 6 logits:
        [s0_empty, s0_full,
         s1_empty, s1_full,
         s2_empty, s2_full]
    """

    def __init__(self, num_spots: int = NUM_SPOTS, num_classes: int = 2):
        super().__init__()
        self.num_spots = num_spots
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 64x64 -> two pools -> 16x16
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_spots * num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (B,16,32,32)
        x = self.pool(F.relu(self.conv2(x)))   # (B,32,16,16)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                        # (B, 6)
        return x


inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


def load_model(device=None):
    if device is None:
        device = torch.device("cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FILENAME)

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file '{MODEL_FILENAME}' not found at: {model_path}")
        sys.exit(1)

    model = SimplePatternNetBinary(num_spots=NUM_SPOTS)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print(f"[OK] Loaded model from {model_path} on {device}")
    return model, device


def predict_empty_spot_from_frame(frame_rgb, model, device):
    """
    frame_rgb: numpy array (H,W,3) in RGB from Camera.get_frame()

    Returns:
        preds: np.array of shape (NUM_SPOTS,), each in {0,1}
        empty_indices: list of 0-based indices where spot is empty
        chosen_spot: int
            - 0  -> no empty spot
            - 1,2,3 -> chosen spot (lowest-numbered empty)
    """
    if frame_rgb is None:
        print("[ERROR] frame_rgb is None")
        return None, [], 0

    frame_rgb = np.asarray(frame_rgb)
    pil_img = Image.fromarray(frame_rgb)
    x = inference_transform(pil_img).unsqueeze(0).to(device)  # (1,3,H,W)

    with torch.no_grad():
        logits = model(x)                       # (1, NUM_SPOTS*2)
        logits = logits.view(1, NUM_SPOTS, 2)   # (1, spots, classes)
        preds = torch.argmax(logits, dim=-1)    # (1, spots)
        preds = preds.cpu().numpy()[0]          # shape (NUM_SPOTS,)

    empty_indices = [i for i, c in enumerate(preds) if c == 0]

    if not empty_indices:
        chosen_spot = 0
    else:
        chosen_spot = empty_indices[0] + 1  # 1-based index

    return preds, empty_indices, chosen_spot


def main():
    if Camera is None:
        print("[ERROR] Camera module (Camera_1.py / camera_1.py) not found.")
        sys.exit(1)

    # 1) Load model
    model, device = load_model()

    # 2) Init camera
    cam = None
    try:
        cam = Camera()
        cam.start_stream()
        time.sleep(0.5)  # small delay for camera to adjust

        # 3) Capture one frame
        frame = cam.get_frame()
        if frame is None:
            print("[ERROR] Could not get frame from camera.")
            return

        # Optional: save the frame for debugging
        try:
            from datetime import datetime
            import cv2

            script_dir = os.path.dirname(os.path.abspath(__file__))
            debug_dir = os.path.join(script_dir, "debug_captures")
            os.makedirs(debug_dir, exist_ok=True)
            filename = datetime.now().strftime("check_%Y%m%d_%H%M%S.jpg")
            filepath = os.path.join(debug_dir, filename)

            # frame is RGB, convert to BGR for OpenCV
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, bgr)
            print(f"[INFO] Saved debug image to: {filepath}")
        except Exception as e:
            print(f"[WARN] Could not save debug image: {e}")

        # 4) Run prediction
        preds, empties, chosen = predict_empty_spot_from_frame(frame, model, device)
        if preds is None:
            print("[ERROR] Prediction failed.")
            return

        # 5) Print results
        print("\n================ RESULT ================\n")
        print(f"Per-spot prediction (0=empty, 1=full): {preds}")
        if empties:
            # convert 0-based indices to 1-based for human-friendly output
            empties_1based = [i + 1 for i in empties]
            print(f"Empty spots (1-based): {empties_1based}")
        else:
            print("Empty spots: none")

        if chosen == 0:
            print("\nAutopilot choice: NO SPOT (autopark would abort)\n")
        else:
            print(f"\nAutopilot choice: SPOT {chosen}\n")

        print("=======================================\n")

    finally:
        # Clean up camera
        if cam is not None:
            try:
                if hasattr(cam, "stop_stream"):
                    cam.stop_stream()
                if hasattr(cam, "close"):
                    cam.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
