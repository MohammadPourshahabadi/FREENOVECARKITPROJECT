#!/usr/bin/env python3
"""
Camera helper for Freenove 4WD car (Picamera2-based).

Provides:
    - Camera.start_image()
    - Camera.save_image(filename)
    - Camera.start_stream(filename=None)
    - Camera.stop_stream()
    - Camera.get_frame()
    - Camera.save_video(filename, duration)
    - Camera.close()
"""

import time
import io
from threading import Condition

from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder, JpegEncoder
from picamera2.outputs import FileOutput
from libcamera import Transform


class StreamingOutput(io.BufferedIOBase):
    """Simple buffer that always holds the latest JPEG frame."""

    def __init__(self):
        super().__init__()
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        # Called by Picamera2 when a new JPEG frame is ready
        with self.condition:
            self.frame = buf
            self.condition.notify_all()
        return len(buf)


class Camera:
    def __init__(
        self,
        preview_size=(640, 480),
        stream_size=(400, 300),
        hflip=False,
        vflip=False,
    ):
        """
        Initialize camera object.

        preview_size: size used for local preview (QTGL / DRM)
        stream_size : size used for JPEG streaming frames
        """
        self.picam2 = Picamera2()
        self.transform = Transform(
            hflip=1 if hflip else 0,
            vflip=1 if vflip else 0,
        )

        # Preview configuration
        self.preview_config = self.picam2.create_preview_configuration(
            main={"size": preview_size},
            transform=self.transform,
        )
        self.picam2.configure(self.preview_config)

        # Streaming configuration
        self.stream_size = stream_size
        self.stream_config = self.picam2.create_video_configuration(
            main={"size": stream_size},
            transform=self.transform,
        )

        self.output = StreamingOutput()
        self.streaming = False

    # ---------- Simple preview & snapshot ----------

    def start_image(self, use_qtgl=True):
        """
        Show a live preview window.

        On a desktop/VNC session, use_qtgl=True is fine.
        On a pure console (no X), set use_qtgl=False to use DRM instead.
        """
        if use_qtgl:
            self.picam2.start_preview(Preview.QTGL)
        else:
            self.picam2.start_preview(Preview.DRM)

        # Ensure it's using preview config
        self.picam2.configure(self.preview_config)
        self.picam2.start()

    def save_image(self, filename: str):
        """
        Capture a still image to filename.
        Returns metadata dict or None on error.
        """
        try:
            if not self.picam2.started:
                self.picam2.start()
                time.sleep(0.5)
            return self.picam2.capture_file(filename)
        except Exception as e:
            print(f"[Camera] Error capturing image: {e}")
            return None

    # ---------- Streaming for dashboard / video ----------

    def start_stream(self, filename: str = None):
        """
        Start video stream.

        - If filename is given: record H.264 video to that file.
        - If no filename: stream JPEG frames into self.output
          so get_frame() can be used by the GUI.
        """
        if self.streaming:
            return

        # Switch to stream configuration
        if self.picam2.started:
            self.picam2.stop()
        self.picam2.configure(self.stream_config)

        if filename:
            encoder = H264Encoder()
            output = FileOutput(filename)
        else:
            encoder = JpegEncoder()
            output = FileOutput(self.output)

        self.picam2.start_recording(encoder, output)
        self.streaming = True

    def stop_stream(self):
        """Stop streaming / recording if active."""
        if not self.streaming:
            return
        try:
            self.picam2.stop_recording()
        except Exception as e:
            print(f"[Camera] Error stopping stream: {e}")
        self.streaming = False

    def get_frame(self) -> bytes:
        """
        Return latest JPEG frame from StreamingOutput.
        Used by the Tkinter dashboard to update the preview.
        """
        if not self.streaming:
            return None
        with self.output.condition:
            self.output.condition.wait()
            return self.output.frame

    def save_video(self, filename: str, duration: int = 10):
        """Record an H.264 video for 'duration' seconds."""
        self.start_stream(filename)
        time.sleep(duration)
        self.stop_stream()

    # ---------- Cleanup ----------

    def close(self):
        """Release resources cleanly."""
        if self.streaming:
            self.stop_stream()
        try:
            self.picam2.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Simple self-test when run directly:
    print("Camera self-test: preview + snapshot")

    cam = Camera()

    # Show a 5s preview (QTGL, so run from desktop/VNC, not bare SSH)
    try:
        cam.start_image(use_qtgl=True)
        time.sleep(5)
    except Exception as e:
        print(f"[Camera] Preview error (try use_qtgl=False if headless): {e}")

    # Capture test image
    meta = cam.save_image("test_image.jpg")
    print("Capture metadata:", meta)

    cam.close()
    print("Done.")
