#!/usr/bin/env python3
import tkinter as tk
import sys
import subprocess
import os

from motor import Ordinary_Car

# Sensor modules from the same folder
# Ultrasonic.py should define class Ultrasonic with get_distance()
# infrared.py should define class Infrared with read_all_infrared()
try:
    from Ultrasonic import Ultrasonic
except ImportError:
    Ultrasonic = None

try:
    from infrared import Infrared
except ImportError:
    Infrared = None

SPEED_MIN = 0
SPEED_MAX = 4095
DEFAULT_SPEED = 1100  # your default speed

KEY_BG = "#eeeeee"
KEY_ACTIVE_BG = "#cccccc"
STOP_BG = "#ff6666"
STOP_ACTIVE_BG = "#ff3333"


class ManualHoldDashboard:
    def __init__(self, master):
        self.master = master
        self.master.title("Freenove 4WD - Hold-to-Move Dashboard")

        # --- Core hardware objects ---
        self.car = Ordinary_Car()
        self.pressed = set()

        # External process handles
        self.camera_process = None
        self.autopark_process = None

        # Sensors
        self.ultrasonic = Ultrasonic() if Ultrasonic is not None else None
        self.infrared = Infrared() if Infrared is not None else None

        # Speed
        self.speed = tk.IntVar(value=DEFAULT_SPEED)

        # ========== UI HEADER ==========
        title = tk.Label(
            master,
            text="Freenove 4WD Manual Control (Hold to Move)",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=(8, 2))

        subtitle = tk.Label(
            master,
            text=(
                "Hold W/A/S/D to move  |  SPACE = Stop\n"
                "Mouse: press key = Move, release = Stop\n"
                "Camera button / C = toggle camera.py\n"
                "Auto Park button / P = toggle autopark.py"
            ),
            font=("Arial", 10)
        )
        subtitle.pack(pady=(0, 8))

        # ========== SPEED SLIDER ==========
        speed_frame = tk.Frame(master)
        speed_frame.pack(pady=5)
        tk.Label(speed_frame, text="Speed", font=("Arial", 11)).pack(side=tk.LEFT, padx=(0, 6))
        self.speed_scale = tk.Scale(
            speed_frame,
            from_=SPEED_MIN,
            to=SPEED_MAX,
            orient=tk.HORIZONTAL,
            variable=self.speed,
            length=260
        )
        self.speed_scale.pack(side=tk.LEFT)

        # ========== ON-SCREEN KEYS ==========
        kb_frame = tk.Frame(master)
        kb_frame.pack(pady=10)

        def make_key(text, row, col, press_cmd, release_cmd,
                     w=6, bg=KEY_BG, abg=KEY_ACTIVE_BG):
            btn = tk.Button(
                kb_frame,
                text=text,
                width=w,
                height=2,
                font=("Arial", 14, "bold"),
                bg=bg,
                activebackground=abg
            )
            btn.grid(row=row, column=col, padx=5, pady=5)
            btn.bind("<ButtonPress-1>", lambda e: press_cmd())
            btn.bind("<ButtonRelease-1>", lambda e: release_cmd())
            return btn

        # Layout:
        #    [   ][ W ][   ]
        #    [ A ][ S ][ D ]
        self.btn_w = make_key("W", 0, 1, self._press_forward,  self._release_dir)
        self.btn_a = make_key("A", 1, 0, self._press_left,     self._release_dir)
        self.btn_s = make_key("S", 1, 1, self._press_backward, self._release_dir)
        self.btn_d = make_key("D", 1, 2, self._press_right,    self._release_dir)

        # STOP
        self.btn_stop = tk.Button(
            master,
            text="STOP",
            width=10,
            height=2,
            font=("Arial", 14, "bold"),
            bg=STOP_BG,
            fg="white",
            activebackground=STOP_ACTIVE_BG,
            command=self.stop
        )
        self.btn_stop.pack(pady=(0, 6))

        # CAMERA
        self.btn_camera = tk.Button(
            master,
            text="Camera",
            width=10,
            height=1,
            font=("Arial", 11, "bold"),
            command=self.toggle_camera
        )
        self.btn_camera.pack(pady=(0, 4))

        # AUTO PARK
        self.btn_autopark = tk.Button(
            master,
            text="Auto Park",
            width=10,
            height=1,
            font=("Arial", 11, "bold"),
            command=self.toggle_autopark
        )
        self.btn_autopark.pack(pady=(0, 6))

        # ========== SENSOR DISPLAY ==========
        sensor_frame = tk.Frame(master)
        sensor_frame.pack(pady=(4, 8))

        self.ultra_label = tk.Label(
            sensor_frame,
            text="Ultrasonic: --.- cm",
            font=("Arial", 11)
        )
        self.ultra_label.grid(row=0, column=0, padx=10)

        self.ir_label = tk.Label(
            sensor_frame,
            text="Infrared: ---",
            font=("Arial", 11)
        )
        self.ir_label.grid(row=0, column=1, padx=10)

        # Status line
        self.status = tk.Label(master, text="Status: STOPPED", font=("Arial", 11))
        self.status.pack(pady=(0, 8))

        # ========== KEYBOARD BINDINGS ==========
        for key in ("w", "a", "s", "d", "W", "A", "S", "D"):
            self.master.bind(f"<KeyPress-{key}>", self.on_key_press)
            self.master.bind(f"<KeyRelease-{key}>", self.on_key_release)

        self.master.bind("<space>", self.on_space)
        self.master.bind("<c>", lambda e: self.toggle_camera())
        self.master.bind("<C>", lambda e: self.toggle_camera())
        self.master.bind("<p>", lambda e: self.toggle_autopark())
        self.master.bind("<P>", lambda e: self.toggle_autopark())

        # Close handler
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

        # Start periodic sensor update
        self.master.after(200, self.update_sensors)

        # Focus for keys
        self.master.focus_force()

    # ===== Helpers =====
    def _get_speed(self):
        s = self.speed.get()
        if s < SPEED_MIN:
            s = SPEED_MIN
        if s > SPEED_MAX:
            s = SPEED_MAX
        return s

    def _is_camera_running(self):
        return self.camera_process is not None and self.camera_process.poll() is None

    def _is_autopark_running(self):
        return self.autopark_process is not None and self.autopark_process.poll() is None

    # ===== Manual drive: keyboard =====
    def on_key_press(self, event):
        if self._is_autopark_running():
            self.status.config(text="Status: AUTOPARK RUNNING (manual disabled)")
            return
        key = event.keysym.lower()
        if key in ("w", "a", "s", "d"):
            if key not in self.pressed:
                self.pressed.add(key)
                self.update_motion()

    def on_key_release(self, event):
        if self._is_autopark_running():
            return
        key = event.keysym.lower()
        if key in self.pressed:
            self.pressed.remove(key)
            self.update_motion()

    def on_space(self, event):
        if self._is_autopark_running():
            self.status.config(text="Status: AUTOPARK RUNNING (stop with Auto Park button)")
            return
        self.pressed.clear()
        self.stop()

    # ===== Manual drive: mouse (on-screen keys) =====
    def _press_forward(self):
        if self._is_autopark_running():
            self.status.config(text="Status: AUTOPARK RUNNING (manual disabled)")
            return
        self.pressed = {"w"}
        self.update_motion()

    def _press_backward(self):
        if self._is_autopark_running():
            self.status.config(text="Status: AUTOPARK RUNNING (manual disabled)")
            return
        self.pressed = {"s"}
        self.update_motion()

    def _press_left(self):
        if self._is_autopark_running():
            self.status.config(text="Status: AUTOPARK RUNNING (manual disabled)")
            return
        self.pressed = {"a"}
        self.update_motion()

    def _press_right(self):
        if self._is_autopark_running():
            self.status.config(text="Status: AUTOPARK RUNNING (manual disabled)")
            return
        self.pressed = {"d"}
        self.update_motion()

    def _release_dir(self):
        if self._is_autopark_running():
            return
        self.pressed.clear()
        self.update_motion()

    # ===== Manual motion logic =====
    def update_motion(self):
        if self._is_autopark_running():
            self.status.config(text="Status: AUTOPARK RUNNING (manual disabled)")
            return

        s = self._get_speed()

        if not self.pressed:
            self.stop()
            return

        f = 'w' in self.pressed
        b = 's' in self.pressed
        l = 'a' in self.pressed
        r = 'd' in self.pressed

        if f and not b:
            if l and not r:
                self.car.set_motor_model(int(0.4 * s), s, s, s)
                self.status.config(text=f"Status: FORWARD-LEFT  speed={s}")
            elif r and not l:
                self.car.set_motor_model(s, s, int(0.4 * s), s)
                self.status.config(text=f"Status: FORWARD-RIGHT speed={s}")
            else:
                self.car.set_motor_model(s, s, s, s)
                self.status.config(text=f"Status: FORWARD       speed={s}")

        elif b and not f:
            if l and not r:
                self.car.set_motor_model(int(-0.4 * s), -s, -s, -s)
                self.status.config(text=f"Status: BACKWARD-LEFT  speed={s}")
            elif r and not l:
                self.car.set_motor_model(-s, -s, int(-0.4 * s), -s)
                self.status.config(text=f"Status: BACKWARD-RIGHT speed={s}")
            else:
                self.car.set_motor_model(-s, -s, -s, -s)
                self.status.config(text=f"Status: BACKWARD       speed={s}")
        else:
            if l and not r:
                self.car.set_motor_model(-s, -s, s, s)
                self.status.config(text=f"Status: LEFT PIVOT      speed={s}")
            elif r and not l:
                self.car.set_motor_model(s, s, -s, -s)
                self.status.config(text=f"Status: RIGHT PIVOT     speed={s}")
            else:
                self.stop()

    def stop(self):
        self.car.set_motor_model(0, 0, 0, 0)
        if not self._is_autopark_running():
            self.status.config(text="Status: STOPPED")

    # ===== Sensors: periodic update =====
    def update_sensors(self):
        # Ultrasonic distance
        if self.ultrasonic is not None:
            try:
                dist = self.ultrasonic.get_distance()
                if dist is not None:
                    self.ultra_label.config(text=f"Ultrasonic: {dist:.1f} cm")
                else:
                    self.ultra_label.config(text="Ultrasonic: --.- cm")
            except Exception:
                self.ultra_label.config(text="Ultrasonic: ERROR")

        else:
            self.ultra_label.config(text="Ultrasonic: (module not found)")

        # Infrared sensors
        if self.infrared is not None:
            try:
                val = self.infrared.read_all_infrared()
                # Show binary (3 bits) + decimal
                self.ir_label.config(text=f"Infrared: {val:03b} ({val})")
            except Exception:
                self.ir_label.config(text="Infrared: ERROR")
        else:
            self.ir_label.config(text="Infrared: (module not found)")

        # Schedule next update
        self.master.after(200, self.update_sensors)

    # ===== Camera control =====
    def toggle_camera(self):
        if not self._is_camera_running():
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                self.camera_process = subprocess.Popen(
                    ["python3", "camera.py"],
                    cwd=script_dir
                )
                self.status.config(text="Status: CAMERA STARTED")
                self.btn_camera.config(text="Camera (On)")
            except Exception as e:
                self.status.config(text=f"Camera error: {e}")
        else:
            self._stop_camera_process()
            if not self._is_autopark_running():
                self.status.config(text="Status: CAMERA STOPPED")
            self.btn_camera.config(text="Camera")

    def _stop_camera_process(self):
        if self.camera_process is not None:
            try:
                self.camera_process.terminate()
            except Exception:
                pass
            self.camera_process = None

    # ===== Auto Park control =====
    def toggle_autopark(self):
        if not self._is_autopark_running():
            try:
                # Clear manual state & stop
                self.pressed.clear()
                self.stop()
                script_dir = os.path.dirname(os.path.abspath(__file__))
                self.autopark_process = subprocess.Popen(
                    ["python3", "autopark.py"],
                    cwd=script_dir
                )
                self.status.config(text="Status: AUTOPARK RUNNING")
                self.btn_autopark.config(text="Auto Park (On)")
            except Exception as e:
                self.status.config(text=f"Autopark error: {e}")
        else:
            self._stop_autopark_process()
            if not self._is_camera_running():
                self.status.config(text="Status: AUTOPARK STOPPED")
            self.btn_autopark.config(text="Auto Park")

    def _stop_autopark_process(self):
        if self.autopark_process is not None:
            try:
                self.autopark_process.terminate()
            except Exception:
                pass
            self.autopark_process = None

    # ===== Cleanup =====
    def on_close(self):
        try:
            self.pressed.clear()
            self.stop()
            self._stop_camera_process()
            self._stop_autopark_process()

            # Clean sensors if they expose close()
            if self.ultrasonic is not None and hasattr(self.ultrasonic, "close"):
                self.ultrasonic.close()
            if self.infrared is not None and hasattr(self.infrared, "close"):
                self.infrared.close()

            if hasattr(self.car, "close"):
                self.car.close()
        except Exception as e:
            print("Cleanup error:", e)

        self.master.destroy()


def main():
    root = tk.Tk()
    app = ManualHoldDashboard(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()
        sys.exit(0)


if __name__ == "__main__":
    main()
