#!/usr/bin/env python3
"""
autopark.py — FINAL VERSION

Features:
- Fixed inverted motor directions (forward = forward, left = left)
- Separate turn times for left/right (asymmetric calibration)
- Safe low-speed mode for table testing
- Assumes car is manually placed ~60 cm from spots
"""

import os
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# ---- Hardware modules ----
from motor import Ordinary_Car

try:
    from ultrasonic import Ultrasonic
except ImportError:
    try:
        from Ultrasonic import Ultrasonic
    except ImportError:
        Ultrasonic = None

try:
    from Camera_1 import Camera
except ImportError:
    try:
        from camera_1 import Camera
    except ImportError:
        Camera = None


# ================== SPEED & TURN CONFIG ==================

# 🚗 HIGH-SPEED MODE (use only in open area, NOT near edges)
# AUTO_SPEED = 1100
# TURN_TIME_LEFT_90 = 1.5
# TURN_TIME_RIGHT_90 = 1.7
# FORWARD_10CM_TIME = 0.35
# FINAL_PARK_TIME = 1.4

# 🐢 LOW-SPEED MODE (recommended for tables — START HERE)
AUTO_SPEED = 700
TURN_TIME_LEFT_90 = 0.8   # Tune: increase if <90°, decrease if >90°
TURN_TIME_RIGHT_90 = 0.775  # Right often weaker → needs more time
FORWARD_10CM_TIME = 0.7
FINAL_PARK_TIME = 2.0


# ================== FIXED CONFIG ==================
NUM_SPOTS = 3
IMAGE_SIZE = 64


# ================== MODEL ==================

class SimplePatternNetBinary(nn.Module):
    def __init__(self, num_spots=NUM_SPOTS, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_spots * num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


def load_model(device=None):
    if device is None:
        device = torch.device("cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "parking_model.pt")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)
    model = SimplePatternNetBinary()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[OK] Model loaded | Speed: {AUTO_SPEED}")
    return model, device


def predict_empty_spot_from_frame(frame_rgb, model, device):
    if frame_rgb is None:
        return 0
    x = inference_transform(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).view(1, 3, 2)
        preds = torch.argmax(logits, dim=-1)[0].cpu().numpy()
    empty = [i for i, c in enumerate(preds) if c == 0]
    return empty[0] + 1 if empty else 0


# ================== MOVEMENT (CORRECTED FOR INVERTED MOTORS) ==================

def stop_car(car):
    car.set_motor_model(0, 0, 0, 0)
    time.sleep(0.1)

def turn_left_90(car, speed=2000):
    """Physically turn LEFT (uses corrected motor signs)"""
    print(f"[ACTION] LEFT turn - {TURN_TIME_LEFT_90:.2f}s")
    # This pattern turns LEFT on your car (confirmed working)
    car.set_motor_model(-speed, -speed, speed, speed)
    time.sleep(TURN_TIME_LEFT_90)
    stop_car(car)

def turn_right_90(car, speed=2000):
    """Physically turn RIGHT"""
    print(f"[ACTION] RIGHT turn - {TURN_TIME_RIGHT_90:.2f}s")
    # This pattern turns RIGHT on your car
    car.set_motor_model(speed, speed, -speed, -speed)
    time.sleep(TURN_TIME_RIGHT_90)
    stop_car(car)

def drive_forward_10cm(car, speed=AUTO_SPEED):
    """Move forward ~10 cm (corrected direction)"""
    print(f"[ACTION] Forward 10cm - {FORWARD_10CM_TIME:.2f}s")
    car.set_motor_model(-speed, -speed, -speed, -speed)  # inverted = forward
    time.sleep(FORWARD_10CM_TIME)
    stop_car(car)

def final_park_forward(car, speed=AUTO_SPEED):
    """Final move into parking spot"""
    print(f"[ACTION] Final park - {FINAL_PARK_TIME:.2f}s")
    car.set_motor_model(-speed, -speed, -speed, -speed)
    time.sleep(FINAL_PARK_TIME)
    stop_car(car)


# ================== MAIN ==================

def main():
    if Camera is None:
        print("[ERROR] Camera module (Camera_1.py) not found.")
        sys.exit(1)

    car = Ordinary_Car()
    camera = None
    ultrasonic = Ultrasonic() if Ultrasonic else None

    try:
        # Initialize camera
        camera = Camera()
        camera.start_stream()
        time.sleep(0.5)

        # Optional distance check
        if ultrasonic:
            d = ultrasonic.get_distance()
            if d is not None:
                print(f"[INFO] Initial distance: {d:.1f} cm")

        # Load model and capture frame
        model, device = load_model()
        frame = camera.get_frame()
        if frame is None:
            print("[ERROR] Failed to capture camera frame")
            return

        # Predict spot
        spot = predict_empty_spot_from_frame(frame, model, device)
        print(f"[INFO] Chosen parking spot: {spot}")
        if spot == 0:
            print("[INFO] No empty spot. Autopark aborted.")
            return

        # Execute parking sequence
        if spot == 1:
            print("→ Parking in LEFT spot (1)")
            turn_left_90(car)
            drive_forward_10cm(car)
            turn_right_90(car)
            final_park_forward(car)

        elif spot == 2:
            print("→ Parking in MIDDLE spot (2)")
            final_park_forward(car)

        elif spot == 3:
            print("→ Parking in RIGHT spot (3)")
            turn_right_90(car)
            drive_forward_10cm(car)
            turn_left_90(car)
            final_park_forward(car)

        print("[OK] Autopark completed successfully!")

    except KeyboardInterrupt:
        print("\n[INFO] Autopark interrupted by user.")
    finally:
        stop_car(car)
        if camera:
            try:
                camera.stop_stream()
                if hasattr(camera, 'close'):
                    camera.close()
            except:
                pass


if __name__ == "__main__":
    main()