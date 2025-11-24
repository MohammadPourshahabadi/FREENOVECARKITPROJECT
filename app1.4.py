#!/usr/bin/env python3
from flask import Flask, render_template, request, jsonify, Response
import threading
import time
import os
from datetime import datetime
import cv2
import logging

logging.basicConfig(level=logging.INFO)

# Hardware imports
from motor import Ordinary_Car
try:
    from ultrasonic import Ultrasonic
except ImportError:
    try:
        from Ultrasonic import Ultrasonic
    except ImportError:
        Ultrasonic = None

try:
    from infrared import Infrared
except ImportError:
    Infrared = None

try:
    from Camera_1 import Camera
except ImportError:
    Camera = None

try:
    from servo import Servo
except ImportError:
    Servo = None
    logging.error("servo.py not found.")
    pca9685_servo = None

# Autopark import
try:
    from autopark_logic import run_autopark
    AUTOPARK_AVAILABLE = True
except Exception as e:
    logging.error(f"Failed to load autopark_logic: {e}")
    AUTOPARK_AVAILABLE = False

app = Flask(__name__)

# ----------------- Global State -----------------
car = Ordinary_Car()
ultrasonic = None
infrared = None
camera = None

ultra_enabled = False
ir_enabled = False
camera_enabled = False

camera_lock = threading.Lock()
DEFAULT_SPEED = 1500
speed = DEFAULT_SPEED
status_msg = "STOPPED"

capture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Captured_images")
os.makedirs(capture_dir, exist_ok=True)

# --- Servo ---
PAN_SERVO_CHANNEL = '0'
TILT_SERVO_CHANNEL = '1'
pca9685_servo = None
if Servo is not None:
    try:
        pca9685_servo = Servo()
        logging.info("PCA9685 initialized.")
    except Exception as e:
        logging.error(f"PCA9685 init failed: {e}")

DEFAULT_PAN_ANGLE = 90.0
DEFAULT_TILT_ANGLE = 0.0
current_pan_angle = DEFAULT_PAN_ANGLE
current_tilt_angle = DEFAULT_TILT_ANGLE

if pca9685_servo is not None:
    try:
        pca9685_servo.set_servo_pwm(PAN_SERVO_CHANNEL, DEFAULT_PAN_ANGLE)
        pca9685_servo.set_servo_pwm(TILT_SERVO_CHANNEL, DEFAULT_TILT_ANGLE)
    except Exception as e:
        logging.error(f"Set default servo angles failed: {e}")

# Autopark state
_autopark_running = False
_autopark_lock = threading.Lock()


# ----------------- Helper Functions -----------------
def send_command(cmd):
    global status_msg, car, speed, current_pan_angle, current_tilt_angle
    global _autopark_running, camera_enabled, ultrasonic

    cmd = cmd.upper()

    # --- Servo ---
    if cmd.startswith("SET_PAN_ANGLE "):
        if pca9685_servo is None:
            status_msg = "Servo not initialized."
            return
        try:
            angle = float(cmd.split(" ", 1)[1])
            if 0 <= angle <= 180:
                pca9685_servo.set_servo_pwm(PAN_SERVO_CHANNEL, angle)
                current_pan_angle = angle
                status_msg = f"Pan set to {angle}°"
            else:
                status_msg = f"Invalid pan angle: {angle}"
        except (ValueError, IndexError):
            status_msg = f"Invalid SET_PAN_ANGLE: {cmd}"

    elif cmd.startswith("SET_TILT_ANGLE "):
        if pca9685_servo is None:
            status_msg = "Servo not initialized."
            return
        try:
            angle = float(cmd.split(" ", 1)[1])
            if 0 <= angle <= 180:
                pca9685_servo.set_servo_pwm(TILT_SERVO_CHANNEL, angle)
                current_tilt_angle = angle
                status_msg = f"Tilt set to {angle}°"
            else:
                status_msg = f"Invalid tilt angle: {angle}"
        except (ValueError, IndexError):
            status_msg = f"Invalid SET_TILT_ANGLE: {cmd}"

    # --- Motor (Reversed) ---
    elif cmd == "FORWARD":
        car.set_motor_model(-speed, -speed, -speed, -speed)
        status_msg = f"FORWARD speed={speed}"
    elif cmd == "BACKWARD":
        car.set_motor_model(speed, speed, speed, speed)
        status_msg = f"BACKWARD speed={speed}"
    elif cmd == "LEFT":
        car.set_motor_model(speed, speed, -speed, -speed)
        status_msg = f"LEFT pivot speed={speed}"
    elif cmd == "RIGHT":
        car.set_motor_model(-speed, -speed, speed, speed)
        status_msg = f"RIGHT pivot speed={speed}"
    elif cmd == "STOP":
        car.set_motor_model(0, 0, 0, 0)
        status_msg = "STOPPED"

    # --- Speed ---
    elif cmd.startswith("SET_SPEED "):
        try:
            new_speed = int(cmd.split(" ", 1)[1])
            if 0 <= new_speed <= 4095:
                speed = new_speed
                status_msg = f"Speed set to {speed}"
            else:
                status_msg = f"Invalid speed: {new_speed}"
        except (ValueError, IndexError):
            status_msg = f"Invalid SET_SPEED: {cmd}"

    # --- Camera/Sensors ---
    elif cmd == "CAPTURE IMAGE":
        capture_image()
    elif cmd == "CAMERA ON":
        toggle_camera(True)
    elif cmd == "CAMERA OFF":
        toggle_camera(False)
    elif cmd == "ULTRASONIC ON":
        toggle_ultrasonic(True)
    elif cmd == "ULTRASONIC OFF":
        toggle_ultrasonic(False)

    # --- Autopark ---
    elif cmd == "AUTOPARK ON":
        if not AUTOPARK_AVAILABLE:
            status_msg = "Autopark: Module not available"
            return

        if _autopark_running:
            status_msg = "Autopark: Already running"
            return

        def _run_autopark():
            global _autopark_running, camera_enabled
            try:
                _autopark_running = True
                status_msg = "Autopark: Starting..."

                original_camera_enabled = camera_enabled
                if camera_enabled and camera:
                    with camera_lock:
                        camera_enabled = False
                        time.sleep(0.2)

                run_autopark(camera_instance=camera, ultrasonic_instance=ultrasonic)

                status_msg = "Autopark: Completed"
                logging.info("Autopark finished")

            except Exception as e:
                status_msg = f"Autopark failed: {e}"
                logging.error(f"Autopark error: {e}")
            finally:
                if original_camera_enabled:
                    camera_enabled = True
                _autopark_running = False

        threading.Thread(target=_run_autopark, daemon=True).start()
        status_msg = "Autopark: Started"

    elif cmd == "AUTOPARK OFF":
        status_msg = "Autopark: Cannot be stopped mid-sequence"

    else:
        status_msg = f"Unknown command: {cmd}"

    logging.info(f"Command '{cmd}' → {status_msg}")


# ----------------- Sensor / Camera Toggles -----------------
def toggle_ultrasonic(enable):
    global ultrasonic, ultra_enabled, status_msg
    if enable:
        if ultrasonic is None and Ultrasonic is not None:
            try:
                ultrasonic = Ultrasonic()
                ultra_enabled = True
                status_msg = "Ultrasonic ON"
            except RuntimeError as e:
                ultra_enabled = False
                status_msg = f"Ultrasonic error: {e}"
        elif Ultrasonic is None:
            status_msg = "Ultrasonic module not found"
        else:
            ultra_enabled = True
            status_msg = "Ultrasonic ON"
    else:
        ultra_enabled = False
        status_msg = "Ultrasonic OFF"
        if ultrasonic is not None:
            try:
                ultrasonic.close()
            except Exception:
                pass
            ultrasonic = None


def toggle_infrared(enable):
    global infrared, ir_enabled, status_msg
    if enable:
        if infrared is None and Infrared is not None:
            try:
                infrared = Infrared()
                ir_enabled = True
                status_msg = "Infrared ON"
            except RuntimeError as e:
                ir_enabled = False
                status_msg = f"Infrared error: {e}"
        elif Infrared is None:
            status_msg = "Infrared module not found"
        else:
            ir_enabled = True
            status_msg = "Infrared ON"
    else:
        ir_enabled = False
        status_msg = "Infrared OFF"
        if infrared is not None:
            try:
                infrared.close()
            except Exception:
                pass
            infrared = None


def toggle_camera(enable):
    global camera, camera_enabled, status_msg
    if enable:
        if camera is None and Camera is not None:
            try:
                temp_camera = Camera()
                temp_camera.start_stream()
                camera = temp_camera
                camera_enabled = True
                status_msg = "Camera ON"
            except Exception as e:
                camera_enabled = False
                status_msg = f"Camera error: {e}"
                camera = None
        elif Camera is None:
            status_msg = "Camera module not found"
        else:
            camera_enabled = True
            status_msg = "Camera ON"
    else:
        camera_enabled = False
        status_msg = "Camera OFF"
        if camera is not None:
            try:
                camera.stop_stream()
                camera.close()
            except Exception:
                pass
            camera = None


def capture_image():
    global camera, status_msg
    if Camera is None:
        status_msg = "Camera module not found"
        return

    frame = None
    try:
        if camera_enabled and camera is not None:
            with camera_lock:
                frame = camera.get_frame()
        else:
            status_msg = "Cannot capture: Camera not ON"
            return

        if frame is None:
            status_msg = "No frame captured"
            return

        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        filepath = os.path.join(capture_dir, filename)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, bgr)
        status_msg = f"Image saved: {filename}"

    except Exception as e:
        status_msg = f"Capture error: {e}"


# ----------------- Routes -----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/command", methods=["POST"])
def command():
    try:
        data = request.get_json()
        if not data or 'cmd' not in data:
            return jsonify({"status": "Error: Invalid request"}), 400
        cmd = data.get("cmd")
        send_command(cmd)
        return jsonify({"status": status_msg})
    except Exception as e:
        return jsonify({"status": f"Error: {str(e)}"}), 500


@app.route("/status")
def get_status():
    try:
        ultra = None
        ir = None
        if ultra_enabled and ultrasonic is not None:
            try:
                ultra = ultrasonic.get_distance()
            except Exception:
                ultra = "ERROR"
        if ir_enabled and infrared is not None:
            try:
                ir = infrared.read_all_infrared()
            except Exception:
                ir = "ERROR"
        return jsonify({
            "status": status_msg,
            "ultrasonic": ultra,
            "infrared": ir,
            "pan_servo_angle": current_pan_angle,
            "tilt_servo_angle": current_tilt_angle,
            "autopark_running": _autopark_running
        })
    except Exception as e:
        return jsonify({
            "status": f"Status error: {str(e)}",
            "ultrasonic": None,
            "infrared": None,
            "pan_servo_angle": None,
            "tilt_servo_angle": None,
            "autopark_running": False
        }), 500


@app.route("/autopark_status")
def get_autopark_status():
    """Return latest autopark log messages."""
    try:
        from autopark_logic import get_autopark_messages
        messages = get_autopark_messages()
        return jsonify({"messages": messages, "running": _autopark_running})
    except Exception as e:
        return jsonify({"messages": [f"Error: {e}"], "running": False})


def generate_frames():
    global camera, camera_enabled
    time.sleep(0.2)
    while True:
        if camera_enabled and camera is not None:
            with camera_lock:
                frame = camera.get_frame()
            if frame is not None:
                try:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    ret, buffer = cv2.imencode('.jpg', bgr)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    logging.error(f"Frame encoding error: {e}")
                    break
        else:
            time.sleep(0.1)
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    if not camera_enabled or camera is None:
        return "Camera is OFF", 404
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ----------------- Run -----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)