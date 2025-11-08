#!/usr/bin/env python3
import tkinter as tk
import sys
import subprocess
import os
import io

from motor import Ordinary_Car

# -------- Ultrasonic import (support both names) --------
try:
    from ultrasonic import Ultrasonic
except ImportError:
    try:
        from Ultrasonic import Ultrasonic
    except ImportError:
        Ultrasonic = None

# -------- Infrared import --------
try:
    from infrared import Infrared
except ImportError:
    Infrared = None

# -------- Camera import --------
try:
    from camera import Camera
except ImportError:
    Camera = None

# -------- Pillow for camera preview --------
try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

SPEED_MIN = 0
SPEED_MAX = 4095
DEFAULT_SPEED = 1100

KEY_BG = "#eeeeee"
KEY_ACTIVE_BG = "#cccccc"
STOP_BG = "#ff6666"
STOP_ACTIVE_BG = "#ff3333"


class ManualHoldDashboard:
    def __init__(self, master):
        self.master = master
        self.master.title("Freenove 4WD - Hold-to-Move Dashboard")

        # --- Core state ---
        self.car = Ordinary_Car()
        self.pressed = set()

        # Background processes / devices
        self.autopark_process = None

        self.ultrasonic = None
        self.infrared = None
        self.camera = None

        self.ultra_enabled = False
        self.ir_enabled = False
        self.camera_enabled = False

        self.camera_photo = None  # keep reference

        self.speed = tk.IntVar(value=DEFAULT_SPEED)

        # ---------- Header ----------
        title = tk.Label(
            master,
            text="Freenove 4WD Manual Control (Hold to Move)",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=(8, 2))

        subtitle = tk.Label(
            master,
            text=(
                "Hold W/A/S/D or click keys to move (release = stop)\n"
                "Camera / Ultrasonic / Infrared: switches control live view\n"
                "Auto Park: run autopark.py (manual disabled while ON)"
            ),
            font=("Arial", 10)
        )
        subtitle.pack(pady=(0, 8))

        # ---------- Speed slider ----------
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

        # ---------- On-screen WASD ----------
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

        # ---------- STOP ----------
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

        # ---------- Toggles row ----------
        toggle_frame = tk.Frame(master)
        toggle_frame.pack(pady=(0, 6))

        self.btn_camera = tk.Button(
            toggle_frame,
            text="Camera: OFF",
            width=14,
            font=("Arial", 10, "bold"),
            command=self.toggle_camera
        )
        self.btn_camera.grid(row=0, column=0, padx=4)

        self.btn_ultra = tk.Button(
            toggle_frame,
            text="Ultrasonic: OFF",
            width=16,
            font=("Arial", 10, "bold"),
            command=self.toggle_ultrasonic
        )
        self.btn_ultra.grid(row=0, column=1, padx=4)

        self.btn_ir = tk.Button(
            toggle_frame,
            text="Infrared: OFF",
            width=14,
            font=("Arial", 10, "bold"),
            command=self.toggle_infrared
        )
        self.btn_ir.grid(row=0, column=2, padx=4)

        self.btn_autopark = tk.Button(
            toggle_frame,
            text="Auto Park: OFF",
            width=16,
            font=("Arial", 10, "bold"),
            command=self.toggle_autopark
        )
        self.btn_autopark.grid(row=0, column=3, padx=4)

        # ---------- Sensor labels ----------
        sensor_frame = tk.Frame(master)
        sensor_frame.pack(pady=(4, 4))

        self.ultra_label = tk.Label(sensor_frame, text="Ultrasonic: OFF", font=("Arial", 11))
        self.ultra_label.grid(row=0, column=0, padx=10)

        self.ir_label = tk.Label(sensor_frame, text="Infrared: OFF", font=("Arial", 11))
        self.ir_label.grid(row=0, column=1, padx=10)

        # ---------- Camera preview ----------
        self.camera_frame = tk.Frame(master)
        self.camera_frame.pack(pady=(4, 8))

        self.camera_label = tk.Label(
            self.camera_frame,
            text="Camera: OFF",
            font=("Arial", 10),
            width=50,
            height=10,
            bg="#000000",
            fg="#ffffff"
        )
        self.camera_label.pack()

        # ---------- Status ----------
        self.status = tk.Label(master, text="Status: STOPPED", font=("Arial", 11))
        self.status.pack(pady=(0, 8))

        # ---------- Keyboard bindings ----------
        for key in ("w", "a", "s", "d", "W", "A", "S", "D"):
            self.master.bind(f"<KeyPress-{key}>", self.on_key_press)
            self.master.bind(f"<KeyRelease-{key}>", self.on_key_release)

        self.master.bind("<space>", self.on_space)

        # Shortcuts for toggles
        self.master.bind("<c>", lambda e: self.toggle_camera())
        self.master.bind("<C>", lambda e: self.toggle_camera())
        self.master.bind("<p>", lambda e: self.toggle_autopark())
        self.master.bind("<P>", lambda e: self.toggle_autopark())

        # Clean exit
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

        # Start periodic sensor polling
        self.master.after(200, self.update_sensors)

        # Focus for keys
        self.master.focus_force()

    # ========== Helpers ==========

    def _get_speed(self):
        s = self.speed.get()
        if s < SPEED_MIN:
            s = SPEED_MIN
        if s > SPEED_MAX:
            s = SPEED_MAX
        return s

    def _is_autopark_running(self):
        return self.autopark_process is not None and self.autopark_process.poll() is None

    # ========== Manual drive: keyboard ==========

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
            self.status.config(text="Status: AUTOPARK RUNNING (stop with Auto Park)")
            return
        self.pressed.clear()
        self.stop()

    # ========== Manual drive: on-screen keys ==========

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

    # ========== Manual motion logic ==========

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

    # ========== Sensors: periodic update ==========

    def update_sensors(self):
        # Ultrasonic
        if self.ultra_enabled and self.ultrasonic is not None:
            try:
                d = self.ultrasonic.get_distance()
                if d is not None:
                    self.ultra_label.config(text=f"Ultrasonic: {d:.1f} cm")
                else:
                    self.ultra_label.config(text="Ultrasonic: --.- cm")
            except Exception:
                self.ultra_label.config(text="Ultrasonic: ERROR")
        elif self.ultra_enabled and Ultrasonic is None:
            self.ultra_label.config(text="Ultrasonic: MODULE MISSING")
        else:
            self.ultra_label.config(text="Ultrasonic: OFF")

        # Infrared
        if self.ir_enabled and self.infrared is not None:
            try:
                val = self.infrared.read_all_infrared()
                self.ir_label.config(text=f"Infrared: {val:03b} ({val})")
            except Exception:
                self.ir_label.config(text="Infrared: ERROR")
        elif self.ir_enabled and Infrared is None:
            self.ir_label.config(text="Infrared: MODULE MISSING")
        else:
            self.ir_label.config(text="Infrared: OFF")

        # Reschedule
        self.master.after(200, self.update_sensors)

    # ========== Camera toggle & preview ==========

    def toggle_camera(self):
        if not self.camera_enabled:
            if Camera is None:
                self.status.config(text="Status: Camera module (camera.py) not found")
                return
            if Image is None or ImageTk is None:
                self.status.config(text="Status: Install Pillow for camera preview")
                return

            try:
                self.camera = Camera()
                # Start JPEG streaming; implementation depends on your camera.py
                self.camera.start_stream()
                self.camera_enabled = True
                self.btn_camera.config(text="Camera: ON")
                self.status.config(text="Status: CAMERA ON")
                self.camera_label.config(text="", bg="#000000")
                self.update_camera_view()
            except Exception as e:
                self.status.config(text=f"Status: Camera init error: {e}")
                self.camera = None
                self.camera_enabled = False
        else:
            self.stop_camera()

    def update_camera_view(self):
        if not self.camera_enabled or self.camera is None:
            self.camera_label.config(text="Camera: OFF", image="", bg="#000000", fg="#ffffff")
            self.camera_photo = None
            return

        try:
            frame_bytes = self.camera.get_frame()
            if frame_bytes:
                img = Image.open(io.BytesIO(frame_bytes))
                img = img.resize((400, 300))
                self.camera_photo = ImageTk.PhotoImage(img)
                self.camera_label.config(image=self.camera_photo, text="")
        except Exception:
            self.camera_label.config(text="Camera error", image="")

        if self.camera_enabled:
            self.master.after(60, self.update_camera_view)

    def stop_camera(self):
        self.camera_enabled = False
        if self.camera is not None:
            try:
                if hasattr(self.camera, "stop_stream"):
                    self.camera.stop_stream()
                if hasattr(self.camera, "close"):
                    self.camera.close()
            except Exception:
                pass
            self.camera = None

        self.btn_camera.config(text="Camera: OFF")
        if not self._is_autopark_running():
            self.status.config(text="Status: CAMERA OFF")
        self.camera_label.config(text="Camera: OFF", image="", bg="#000000", fg="#ffffff")
        self.camera_photo = None

    # ========== Ultrasonic toggle ==========

    def toggle_ultrasonic(self):
        if not self.ultra_enabled:
            if Ultrasonic is None:
                self.status.config(text="Status: Ultrasonic module not found")
                return
            if self.ultrasonic is None:
                try:
                    self.ultrasonic = Ultrasonic()
                except Exception as e:
                    self.status.config(text=f"Status: Ultrasonic init error: {e}")
                    return
            self.ultra_enabled = True
            self.btn_ultra.config(text="Ultrasonic: ON")
            self.status.config(text="Status: Ultrasonic ON")
        else:
            self.ultra_enabled = False
            if self.ultrasonic is not None and hasattr(self.ultrasonic, "close"):
                try:
                    self.ultrasonic.close()
                except Exception:
                    pass
            self.ultrasonic = None
            self.btn_ultra.config(text="Ultrasonic: OFF")
            self.status.config(text="Status: Ultrasonic OFF")

    # ========== Infrared toggle ==========

    def toggle_infrared(self):
        if not self.ir_enabled:
            if Infrared is None:
                self.status.config(text="Status: Infrared module not found")
                return
            if self.infrared is None:
                try:
                    self.infrared = Infrared()
                except Exception as e:
                    self.status.config(text=f"Status: Infrared init error: {e}")
                    return
            self.ir_enabled = True
            self.btn_ir.config(text="Infrared: ON")
            self.status.config(text="Status: Infrared ON")
        else:
            self.ir_enabled = False
            if self.infrared is not None and hasattr(self.infrared, "close"):
                try:
                    self.infrared.close()
                except Exception):
                    pass
            self.infrared = None
            self.btn_ir.config(text="Infrared: OFF")
            self.status.config(text="Status: Infrared OFF")

    # ========== Auto Park toggle (subprocess) ==========

    def toggle_autopark(self):
        if not self._is_autopark_running():
            try:
                self.pressed.clear()
                self.stop()
                script_dir = os.path.dirname(os.path.abspath(__file__))
                self.autopark_process = subprocess.Popen(
                    ["python3", "autopark.py"],
                    cwd=script_dir
                )
                self.btn_autopark.config(text="Auto Park: ON")
                self.status.config(text="Status: AUTOPARK RUNNING")
            except Exception as e:
                self.status.config(text=f"Status: Autopark error: {e}")
        else:
            self._stop_autopark()
            self.btn_autopark.config(text="Auto Park: OFF")
            self.status.config(text="Status: AUTOPARK STOPPED")

    def _stop_autopark(self):
        if self.autopark_process is not None:
            try:
                self.autopark_process.terminate()
            except Exception:
                pass
            self.autopark_process = None

    # ========== Cleanup ==========

    def on_close(self):
        try:
            self.pressed.clear()
            self.stop()
            self.stop_camera()
            self._stop_autopark()

            if self.ultrasonic is not None and hasattr(self.ultrasonic, "close"):
                try:
                    self.ultrasonic.close()
                except Exception:
                    pass
            if self.infrared is not None and hasattr(self.infrared, "close"):
                try:
                    self.infrared.close()
                except Exception:
                    pass

            if hasattr(self.car, "close"):
                try:
                    self.car.close()
                except Exception:
                    pass
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
