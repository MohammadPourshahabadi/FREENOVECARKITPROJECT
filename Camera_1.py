from picamera2 import Picamera2
import cv2
import time


class Camera:
    def __init__(self, size=(640, 480)):
        self.picam2 = Picamera2()
        self.config = self.picam2.create_preview_configuration(
            main={"size": size}
        )
        self.picam2.configure(self.config)
        self.started = False

    def start_stream(self):
        """Start the Picamera2 stream (if not already started)."""
        if not self.started:
            self.picam2.start()
            # small delay to let auto-exposure settle (optional)
            time.sleep(0.5)
            self.started = True

    def get_frame(self):
        """
        Capture and return a single frame as a NumPy array (BGR).

        The dashboard will convert this to RGB and show it with Tkinter.
        """
        if not self.started:
            self.start_stream()
        try:
            frame = self.picam2.capture_array()
            return frame   # This is a NumPy array in BGR format
        except Exception:
            return None

    def stop_stream(self):
        """Stop the stream (you can restart it later if needed)."""
        if self.started:
            self.picam2.stop()
            self.started = False

    def close(self):
        """Release camera resources."""
        self.stop_stream()
        # Picamera2 doesn't strictly require an explicit close, but we keep it
        # for symmetry with the dashboard's cleanup.
        # If a future version exposes .close(), you can call it here.
        # self.picam2.close()  # Uncomment if your Picamera2 version supports it.
        pass


# If you run this file directly: show a simple OpenCV preview (like before)
if __name__ == "__main__":
    cam = Camera()
    cam.start_stream()
    print("Camera started. Press 'q' to exit.")

    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                break

            cv2.imshow("Camera Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()
