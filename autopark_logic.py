# autopark_logic.py
"""
Autopark logic with real-time status messaging.
"""

import os
import time
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Hardware modules
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
    Camera = None


# ===== Autopark Status Messages =====
_autopark_messages = []
_autopark_messages_lock = threading.Lock()

def log_autopark_message(msg):
    """Add a message to the autopark log buffer."""
    global _autopark_messages
    timestamp = time.strftime('%H:%M:%S')
    with _autopark_messages_lock:
        _autopark_messages.append(f"[{timestamp}] {msg}")
        if len(_autopark_messages) > 20:
            _autopark_messages.pop(0)

def get_autopark_messages():
    """Get a copy of current messages."""
    with _autopark_messages_lock:
        return _autopark_messages.copy()

def clear_autopark_messages():
    """Clear the message buffer."""
    global _autopark_messages
    with _autopark_messages_lock:
        _autopark_messages.clear()


# ===== CONFIG =====
NUM_SPOTS = 3
IMAGE_SIZE = 64
AUTO_SPEED = 700
TURN_TIME_90 = 0.6
TURN_TIME_RIGHT_90 = 0.75
FORWARD_10CM_TIME = 0.7
FINAL_PARK_TIME = 2.0


# ===== MODEL =====
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
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = SimplePatternNetBinary()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
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


# ===== MOVEMENT =====
def stop_car(car):
    car.set_motor_model(0, 0, 0, 0)
    time.sleep(0.1)

def turn_left_90(car, speed=2000):
    car.set_motor_model(speed, speed, -speed, -speed)
    time.sleep(TURN_TIME_90)
    stop_car(car)

def turn_right_90(car, speed=2000):
    car.set_motor_model(-speed, -speed, speed, speed)
    time.sleep(TURN_TIME_RIGHT_90)
    stop_car(car)

def drive_forward_10cm(car, speed=AUTO_SPEED):
    car.set_motor_model(-speed, -speed, -speed, -speed)
    time.sleep(FORWARD_10CM_TIME)
    stop_car(car)

def final_park_forward(car, speed=AUTO_SPEED):
    car.set_motor_model(-speed, -speed, -speed, -speed)
    time.sleep(FINAL_PARK_TIME)
    stop_car(car)


# ===== MAIN FUNCTION =====
def run_autopark(camera_instance=None, ultrasonic_instance=None):
    """
    Run autopark sequence with real-time status logging.
    """
    clear_autopark_messages()
    log_autopark_message("Starting autopark sequence...")

    car = None
    camera = camera_instance
    ultrasonic = ultrasonic_instance
    cleanup_camera = False
    cleanup_ultrasonic = False

    try:
        if camera is None:
            if Camera is None:
                log_autopark_message("ERROR: Camera module not found")
                return
            camera = Camera()
            camera.start_stream()
            time.sleep(0.5)
            cleanup_camera = True

        if ultrasonic is None and Ultrasonic:
            ultrasonic = Ultrasonic()
            cleanup_ultrasonic = True

        car = Ordinary_Car()

        if ultrasonic:
            d = ultrasonic.get_distance()
            if d is not None:
                log_autopark_message(f"Initial distance: {d:.1f} cm")

        model, device = load_model()
        frame = camera.get_frame()
        if frame is None:
            log_autopark_message("ERROR: No camera frame")
            return

        spot = predict_empty_spot_from_frame(frame, model, device)
        log_autopark_message(f"Chosen parking spot: {spot}")

        if spot == 0:
            log_autopark_message("No empty spots detected. Autopark aborted.")
            return

        if spot == 1:
            log_autopark_message("Parking in LEFT spot (1)")
            turn_left_90(car)
            drive_forward_10cm(car)
            turn_right_90(car)
            final_park_forward(car)
        elif spot == 2:
            log_autopark_message("Parking in MIDDLE spot (2)")
            final_park_forward(car)
        elif spot == 3:
            log_autopark_message("Parking in RIGHT spot (3)")
            turn_right_90(car)
            drive_forward_10cm(car)
            turn_left_90(car)
            final_park_forward(car)

        log_autopark_message("✅ Autopark completed successfully!")

    except Exception as e:
        log_autopark_message(f"ERROR: Autopark failed: {e}")
        raise e
    finally:
        if car:
            stop_car(car)
        if cleanup_camera and camera:
            try:
                camera.stop_stream()
                getattr(camera, 'close', lambda: None)()
            except:
                pass
        if cleanup_ultrasonic and ultrasonic:
            try:
                ultrasonic.close()
            except:
                pass