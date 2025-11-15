from picamera2 import Picamera2
import cv2
import time


class Camera:
    def __init__(self, size=(640, 480)):
        self.picam2 = Picamera2()
        # Force RGB888 so we know exactly what format we get
        self.config = self.picam2.create_preview_configuration(
            main={"size": size, "format": "RGB888"}
        )
        self.picam2.configure(self.config)
        self.started = False

    def start_stream(self):
        """Start the camera stream if not already started."""
        if not self.started:
            self.picam2.start()
            time.sleep(0.3)  # small delay for auto-exposure
            self.started = True

    def get_frame(self):
        """
        Capture a single frame and return it as an RGB numpy array.
        """
        if not self.started:
            self.start_stream()
        try:
            frame = self.picam2.capture_array()  # RGB888
            return frame
        except Exception:
            return None

    def stop_stream(self):
        """Stop the camera stream."""
        if self.started:
            self.picam2.stop()
            self.started = False

    def close(self):
        """Release camera resources."""
        self.stop_stream()
        # Some versions of Picamera2 have close(), some don't
        try:
            self.picam2.close()
        except AttributeError:
            pass


if __name__ == "__main__":
    # Simple test preview using OpenCV
    cam = Camera()
    cam.start_stream()
    print("Camera started. Press 'q' to exit.")
    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                print("No frame received!")
                break

            # Convert RGB -> BGR for OpenCV display
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Camera Preview", bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()
