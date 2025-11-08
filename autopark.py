#!/usr/bin/env python3
"""
autopark.py

Finalized behavior for your scenario:

- There are exactly 4 fixed parking spots in front of the car.
- Auto Park flow:
    1) Drive forward until ultrasonic distance <= APPROACH_DISTANCE.
    2) Stop and capture ONE front image (all 4 spots visible).
    3) Run deep-learning model -> 4 labels: "empty"/"full"/"partial".
    4) If no "empty": print "All spots are full" and exit.
    5) If one or more "empty": pick one and execute a slow parking maneuver
       into that spot, using ultrasonic to avoid collision.

This is a scaffold:
- Movement timings MUST be tuned on your car.
- Deep-learning part expects a pre-trained parking_model.pt.
"""

import os
import sys
import time
import io
from typing import List, Optional

from motor import Ordinary_Car

# ------- Ultrasonic import (support both names) -------
try:
    from ultrasonic import Ultrasonic
except ImportError:
    try:
        from Ultrasonic import Ultrasonic
    except ImportError:
        Ultrasonic = None

# ------- Camera -------
try:
    from camera import Camera
except ImportError:
    Camera = None

# ------- PIL for image handling -------
try:
    from PIL import Image
except ImportError:
    Image = None

# ------- PyTorch for model -------
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
# Pattern model: 4 spots, 3 classes each
# ============================================================

class SimplePatternNet(nn.Module if nn is not None else object):
    """
    Example network:
    - input: resized full frame (3x64x64)
    - output: 4 * 3 logits (for 4 spots, each empty/full/partial)
    Adjust to match your actual training architecture if needed.
    """

    def __init__(self, num_spots=4, num_classes=3):
        if nn is None:
            return
        super().__init__()
        self.num_spots = num_spots
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_spots * num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> (B,16,32,32)
        x = self.pool(F.relu(self.conv2(x)))  # -> (B,32,16,16)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                       # (B, num_spots * num_classes)
        return x


class ParkingPatternModel:
    """
    Wraps the trained model that decides occupancy per spot.

    - Expects a file 'parking_model.pt'.
    - Predicts: list of 4 labels from {"empty", "full", "partial"}.

    You train this ONCE using images that show all 4 spots with all
    combinations (all full, all empty, 2&2, 1&3, etc.)
    and save the state_dict compatible with SimplePatternNet
    or your own architecture.
    """

    LABELS = ["empty", "full", "partial"]

    def __init__(self, model_path: str = "parking_model.pt"):
        self.model_path = model_path

        self.enabled = (
            torch is not None
            and nn is not None
            and transforms is not None
            and os.path.exists(self.model_path)
        )

        if not self.enabled:
            print("[ParkingModel] Deep learning disabled (missing torch/model). Using safe fallback.")
            self.model = None
            return

        self.device = torch.device("cpu")
        self.num_spots = 4
        self.num_classes = len(self.LABELS)

        # Build and load
        self.model = SimplePatternNet(
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
        Returns labels for 4 spots: ["empty"/"full"/"partial", ...] length 4.
        If disabled -> all "full" to avoid unintended moves.
        """
        if image is None or not self.enabled or self.model is None:
            return ["full"] * 4

        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)             # (1, 12)
            logits = logits.view(4, 3)         # (4 spots, 3 classes)
            probs = torch.softmax(logits, dim=1)

        labels = []
        for i in range(4):
            idx = int(torch.argmax(probs[i]).item())
            labels.append(self.LABELS[idx])

        print("[ParkingModel] Spot labels:", labels)
        return labels


# ============================================================
# Auto Park Controller
# ============================================================

class AutoParkController:
    def __init__(self):
        self.car = Ordinary_Car()

        self.ultra = Ultrasonic() if Ultrasonic is not None else None
        if self.ultra is None:
            print("[AutoPark] WARNING: Ultrasonic not available. Autopark unsafe.")

        self.camera = Camera() if Camera is not None else None
        if self.camera is None:
            print("[AutoPark] WARNING: Camera not available. Autopark disabled.")

        self.model = ParkingPatternModel()

        # --- Tunable parameters ---
        self.APPROACH_DISTANCE = 40.0   # cm from row when scanning
        self.FORWARD_SPEED = 900
        self.PARK_SPEED = 600
        self.TURN_SPEED = 800

        self.MIN_FRONT_DIST_CM = 8.0    # safety stop in front
        self.SIDE_MARGIN_CM = 4.0       # conceptual; achieved via slow movement

    # ---------- Basic helpers ----------

    def stop(self):
        self.car.set_motor_model(0, 0, 0, 0)

    def _read_distance(self) -> Optional[float]:
        if self.ultra is None:
            return None
        try:
            return self.ultra.get_distance()
        except Exception:
            return None

    def _too_close_front(self) -> bool:
        d = self._read_distance()
        if d is not None and d < self.MIN_FRONT_DIST_CM:
            print(f"[AutoPark] FRONT TOO CLOSE: {d:.1f} cm -> STOP")
            self.stop()
            return True
        return False

    def drive_forward_until(self, target_cm: float, timeout: float = 6.0):
        """
        Drive forward until ultrasonic <= target_cm or timeout.
        """
        self.car.set_motor_model(
            self.FORWARD_SPEED, self.FORWARD_SPEED,
            self.FORWARD_SPEED, self.FORWARD_SPEED
        )
        start = time.time()
        while time.time() - start < timeout:
            if self._too_close_front():
                break
            d = self._read_distance()
            if d is not None and d <= target_cm:
                break
            time.sleep(0.02)
        self.stop()

    # ---------- Camera & model ----------

    def _capture_frame(self) -> Optional[Image.Image]:
        if self.camera is None or Image is None:
            return None
        try:
            if not getattr(self.camera, "_ap_streaming", False):
                if hasattr(self.camera, "start_stream"):
                    self.camera.start_stream()
                setattr(self.camera, "_ap_streaming", True)
                time.sleep(0.5)

            frame = self.camera.get_frame() if hasattr(self.camera, "get_frame") else None
            if not frame:
                return None

            return Image.open(io.BytesIO(frame)).convert("RGB")
        except Exception as e:
            print("[AutoPark] Error capturing frame:", e)
            return None

    def scan_spots(self) -> List[str]:
        """
        Capture one image, ask model for 4 labels.
        """
        img = self._capture_frame()
        if img is None:
            print("[AutoPark] No frame captured; treating all as full.")
            return ["full"] * 4
        labels = self.model.predict_all(img)
        return labels

    # ---------- Spot choice & maneuver ----------

    def _choose_spot(self, labels: List[str]) -> Optional[int]:
        """
        Choose which empty spot to use.
        Strategy: first (lowest index) "empty" spot.
        """
        empties = [i for i, lab in enumerate(labels) if lab == "empty"]
        if not empties:
            return None
        return min(empties)

    def _park_in_spot(self, index: int):
        """
        Very simple scripted maneuver into chosen spot.
        Assumes spots lie to the RIGHT side of driving lane.

        You MUST tune the durations so the car ends up centered
        with enough clearance (~4 cm) from both sides.
        """
        print(f"[AutoPark] Parking into spot {index}")

        # Step 1: slight forward adjust per spot index
        base = 0.3       # seconds
        step = 0.25      # extra per further spot
        forward_time = base + step * index
        print(f"[AutoPark] Forward adjust: {forward_time:.2f}s")

        self.car.set_motor_model(
            self.PARK_SPEED, self.PARK_SPEED,
            self.PARK_SPEED, self.PARK_SPEED
        )
        start = time.time()
        while time.time() - start < forward_time:
            if self._too_close_front():
                break
            time.sleep(0.02)
        self.stop()

        # Step 2: slow right-turn arc into bay
        print("[AutoPark] Turning into bay...")
        turn_time = 2.0  # tune this
        start = time.time()
        while time.time() - start < turn_time:
            if self._too_close_front():
                break
            # right wheels slower -> curve to right
            self.car.set_motor_model(
                self.PARK_SPEED, self.PARK_SPEED,
                int(0.5 * self.PARK_SPEED), int(0.5 * self.PARK_SPEED)
            )
            time.sleep(0.05)
        self.stop()

        # Step 3: final slow creep forward until near obstacle
        print("[AutoPark] Final slow creep...")
        start = time.time()
        max_creep = 3.0
        while time.time() - start < max_creep:
            if self._too_close_front():
                break
            self.car.set_motor_model(
                self.PARK_SPEED, self.PARK_SPEED,
                self.PARK_SPEED, self.PARK_SPEED
            )
            time.sleep(0.05)
        self.stop()

        print("[AutoPark] Parking attempt complete.")

    # ---------- Main run ----------

    def run(self):
        print("[AutoPark] Auto-park started.")

        # Check prerequisites
        if self.ultra is None or self.camera is None or self.model.model is None:
            print("[AutoPark] Missing ultrasonic/camera/model. Aborting autopark.")
            self.cleanup()
            return

        try:
            # 1) Approach until we are at scan distance
            print("[AutoPark] Approaching scan distance...")
            self.drive_forward_until(self.APPROACH_DISTANCE)

            if self._too_close_front():
                print("[AutoPark] Too close before scanning; abort.")
                self.cleanup()
                return

            # 2) One scan: classify 4 spots
            labels = self.scan_spots()
            print("[AutoPark] Spot labels:", labels)

            # 3) Check empties
            spot_index = self._choose_spot(labels)
            if spot_index is None:
                print("[AutoPark] All spots are full.")
                # You can read this from dashboard by reading stdout if desired.
                self.cleanup()
                return

            print(f"[AutoPark] Selected empty spot {spot_index}. Executing parking.")
            self._park_in_spot(spot_index)

        except KeyboardInterrupt:
            print("[AutoPark] Interrupted by user.")
        finally:
            self.cleanup()

    def cleanup(self):
        print("[AutoPark] Cleanup.")
        try:
            self.stop()
            if self.ultra is not None and hasattr(self.ultra, "close"):
                self.ultra.close()
            if self.camera is not None and hasattr(self.camera, "close"):
                self.camera.close()
            if hasattr(self.car, "close"):
                self.car.close()
        except Exception as e:
            print("[AutoPark] Cleanup error:", e)


def main():
    controller = AutoParkController()
    controller.run()


if __name__ == "__main__":
    main()
