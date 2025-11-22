#!/usr/bin/env python3
"""
autopark.py

Called from ManualHoldDashboard with:

    subprocess.Popen(["python3", "autopark.py"], cwd=script_dir)

Flow:
  1) Move forward/back to get ultrasonic distance between 55 and 65 cm.
  2) Capture a frame from Camera_1 / camera_1 (front view of 3 spots).
  3) Run CNN model (parking_model.pt) to decide which spot is empty:
       - 0: no empty spots
       - 1,2,3: index of empty spot (lowest-numbered empty spot)
  4) Movement:
       if 1: left 90° (like 'A' for TURN_TIME_90), forward 10 cm (like 'W'),
             right 90° (like 'D'), then forward until 8 cm
       if 2: straight forward until 8 cm
       if 3: right 90° (like 'D'), forward 10 cm (like 'W'),
             left 90° (like 'A'), then forward until 8 cm
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


# ================== CONFIG ==================

NUM_SPOTS = 3        # we have 3 parking spots
IMAGE_SIZE = 64      # must match training
AUTO_SPEED = 1100    # motor speed used during autopark

# Here you "hold the key" for this long:
TURN_TIME_90 = 1.5        # seconds to pivot ~90 degrees (like holding A/D)
FORWARD_10CM_TIME = 0.4   # seconds to move ~10 cm (like holding W)


# ================== MODEL ==================

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
    model_path = os.path.join(script_dir, "parking_model.pt")

    if not os.path.exists(model_path):
        print(f"[ERROR] parking_model.pt not found at {model_path}")
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
        0  -> no empty spot
        1,2,3 -> index of empty spot (lowest-numbered empty spot)
    """
    if frame_rgb is None:
        print("[ERROR] predict_empty_spot_from_frame: frame is None")
        return 0

    frame_rgb = np.asarray(frame_rgb)
    pil_img = Image.fromarray(frame_rgb)
    x = inference_transform(pil_img).unsqueeze(0).to(device)  # (1,3,H,W)

    with torch.no_grad():
        logits = model(x)                      # (1, NUM_SPOTS*2)
        logits = logits.view(1, NUM_SPOTS, 2)  # (1, spots, classes)
        preds = torch.argmax(logits, dim=-1)   # (1, spots)
        preds = preds.cpu().numpy()[0]         # shape (NUM_SPOTS,)

    empty_indices = [i for i, c in enumerate(preds) if c == 0]

    print(f"[DEBUG] per-spot predicted classes (0=empty,1=full): {preds}")
    print(f"[DEBUG] empty spots (0-based): {empty_indices}")

    if not empty_indices:
        return 0

    best_spot = empty_indices[0] + 1  # 1-based
    return best_spot


# ================== MOVEMENT HELPERS ==================

def stop_car(car: Ordinary_Car):
    car.set_motor_model(0, 0, 0, 0)


def drive_forward(car: Ordinary_Car, speed=AUTO_SPEED):
    """
    Equivalent to "holding W" in the dashboard (forward).
    """
    car.set_motor_model(speed, speed, speed, speed)


def drive_backward(car: Ordinary_Car, speed=AUTO_SPEED):
    """
    Equivalent to "holding S" in the dashboard (backward).
    """
    car.set_motor_model(-speed, -speed, -speed, -speed)


def turn_left_90(car: Ordinary_Car, speed=AUTO_SPEED):
    """
    Equivalent to "holding A" (left turn) for TURN_TIME_90 seconds:
      left pivot: (-,+) pattern -> (-s, -s, +s, +s)
    """
    print(f"[ACTION] 'A' (left) for {TURN_TIME_90:.2f} s")
    car.set_motor_model(-speed, -speed, speed, speed)
    time.sleep(TURN_TIME_90)
    stop_car(car)


def turn_right_90(car: Ordinary_Car, speed=AUTO_SPEED):
    """
    Equivalent to "holding D" (right turn) for TURN_TIME_90 seconds:
      right pivot: (+,-) pattern -> (+s, +s, -s, -s)
    """
    print(f"[ACTION] 'D' (right) for {TURN_TIME_90:.2f} s")
    car.set_motor_model(speed, speed, -speed, -speed)
    time.sleep(TURN_TIME_90)
    stop_car(car)


def drive_forward_10cm(car: Ordinary_Car, speed=AUTO_SPEED):
    """
    Equivalent to "holding W" (forward) for FORWARD_10CM_TIME seconds.
    You might need to tune this time to get ~10 cm.
    """
    print(f"[ACTION] 'W' (forward) for {FORWARD_10CM_TIME:.2f} s (~10 cm)")
    car.set_motor_model(speed, speed, speed, speed)
    time.sleep(FORWARD_10CM_TIME)
    stop_car(car)


def drive_until_distance(car: Ordinary_Car, ultrasonic: Ultrasonic,
                         target_cm: float, speed=AUTO_SPEED,
                         max_time: float = 10.0):
    """
    Drive forward until ultrasonic distance <= target_cm or timeout.
    This is like "holding W" but with feedback from the sensor.
    """
    print(f"[INFO] drive forward until distance <= {target_cm} cm")
    t0 = time.time()
    while time.time() - t0 < max_time:
        d = ultrasonic.get_distance()
        if d is None:
            print("[WARN] distance is None, retrying...")
            time.sleep(0.05)
            continue

        print(f"[DEBUG] distance={d:.1f} cm")
        if d <= target_cm:
            print("[INFO] reached target distance")
            break

        drive_forward(car, speed)
        time.sleep(0.05)

    stop_car(car)


def move_to_distance_range(car: Ordinary_Car, ultrasonic: Ultrasonic,
                           dist_min: float, dist_max: float,
                           speed=AUTO_SPEED, max_time: float = 15.0):
    """
    Adjust car so that ultrasonic distance is between dist_min and dist_max.

    Similar idea to tapping W/S manually:
      - If distance > dist_max: go forward (W)
      - If distance < dist_min: go backward (S)
    """
    print(f"[INFO] Adjusting distance to be between {dist_min} and {dist_max} cm")
    t0 = time.time()
    while time.time() - t0 < max_time:
        d = ultrasonic.get_distance()
        if d is None:
            print("[WARN] distance is None, retrying...")
            time.sleep(0.05)
            continue

        print(f"[DEBUG] current distance: {d:.1f} cm")

        if dist_min <= d <= dist_max:
            print("[INFO] distance in target range, stopping.")
            stop_car(car)
            return True
        elif d > dist_max:
            drive_forward(car, speed)
        else:
            drive_backward(car, speed)

        time.sleep(0.05)

    print("[WARN] move_to_distance_range: timeout reached.")
    stop_car(car)
    return False


# ================== MAIN AUTOPARK LOGIC ==================

def main():
    if Ultrasonic is None:
        print("[ERROR] Ultrasonic module not found.")
        sys.exit(1)
    if Camera is None:
        print("[ERROR] Camera module (Camera_1.py / camera_1.py) not found.")
        sys.exit(1)

    car = Ordinary_Car()
    ultrasonic = None
    camera = None

    try:
        # Init sensors
        ultrasonic = Ultrasonic()
        camera = Camera()
        camera.start_stream()
        time.sleep(0.5)  # small delay so camera has good frame

        model, device = load_model()

        # 1) Move to 55–65 cm from obstacle
        ok = move_to_distance_range(car, ultrasonic, 55.0, 65.0,
                                    speed=AUTO_SPEED, max_time=20.0)
        if not ok:
            print("[ERROR] Could not reach target distance range. Aborting autopark.")
            return

        # 2) Capture frame
        time.sleep(0.5)  # small pause to let the car stabilize
        frame = camera.get_frame()
        if frame is None:
            print("[ERROR] Could not get frame from camera. Aborting autopark.")
            return

        # 3) Predict empty spot (1,2,3) or 0 if none
        spot = predict_empty_spot_from_frame(frame, model, device)
        print(f"[INFO] predicted empty spot: {spot}")

        if spot == 0:
            print("[INFO] No empty spot detected. Autopark aborted.")
            return

        # 4) Movement according to spot
        if spot == 1:
            print("[INFO] Parking in spot 1 (left)")
            turn_left_90(car, AUTO_SPEED)             # A for TURN_TIME_90
            drive_forward_10cm(car, AUTO_SPEED)       # W for FORWARD_10CM_TIME
            turn_right_90(car, AUTO_SPEED)            # D for TURN_TIME_90
            drive_until_distance(car, ultrasonic, target_cm=8.0,
                                 speed=AUTO_SPEED, max_time=10.0)

        elif spot == 2:
            print("[INFO] Parking in spot 2 (middle)")
            drive_until_distance(car, ultrasonic, target_cm=8.0,
                                 speed=AUTO_SPEED, max_time=10.0)

        elif spot == 3:
            print("[INFO] Parking in spot 3 (right)")
            turn_right_90(car, AUTO_SPEED)            # D for TURN_TIME_90
            drive_forward_10cm(car, AUTO_SPEED)       # W for FORWARD_10CM_TIME
            turn_left_90(car, AUTO_SPEED)             # A for TURN_TIME_90
            drive_until_distance(car, ultrasonic, target_cm=8.0,
                                 speed=AUTO_SPEED, max_time=10.0)

        stop_car(car)
        print("[OK] Autopark sequence completed.")

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt: stopping autopark.")

    finally:
        # cleanup
        try:
            stop_car(car)
        except Exception:
            pass

        if ultrasonic is not None and hasattr(ultrasonic, "close"):
            try:
                ultrasonic.close()
            except Exception:
                pass

        if camera is not None:
            try:
                if hasattr(camera, "stop_stream"):
                    camera.stop_stream()
                if hasattr(camera, "close"):
                    camera.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
