from picamera2 import Picamera2
import cv2
import os
import time
from datetime import datetime


def main():
    picam2 = Picamera2()

    # Use a still configuration at 640x480
    config = picam2.create_still_configuration(main={"size": (640, 480)})
    picam2.configure(config)

    picam2.start()
    # Small delay so exposure can stabilize
    time.sleep(1)

    # Capture a single frame as a NumPy array (BGR)
    frame = picam2.capture_array()

    picam2.stop()

    # Prepare output folder: Captured_images in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "Captured_images")
    os.makedirs(output_dir, exist_ok=True)

    # Create a filename with timestamp
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    filepath = os.path.join(output_dir, filename)

    # Save using OpenCV
    cv2.imwrite(filepath, frame)

    print(f"Saved image to: {filepath}")


if __name__ == "__main__":
    main()
