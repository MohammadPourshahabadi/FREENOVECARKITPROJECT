#!/usr/bin/env python3
import tkinter as tk
import sys
import subprocess
import os
from motor import Ordinary_Car

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

        # Car controller
        self.car = Ordinary_Car()

        # Track held keys for manual drive
        self.pressed = set()

        # Track camera & autopark processes
        self.camera_process = None
        self.autopark_process = None

        # Speed control
        self.speed = tk.IntVar(value=DEFAULT_SPEED)

        # === UI HEADER ===
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
                "C / Camera button = toggle camera.py\n"
                "P / Auto Park button = toggle autopark.py"
            ),
            font=("Arial", 10)
        )
        subtitle.pack(pady=(0, 8))

        # === SPEED SLIDER ===
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

        # === ON-SCREEN "KEYBOARD" ===
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

        # STOP button
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

        # CAMERA button
        self.btn_camera = tk.Button(
            master,
            text="Camera",
            width=10,
            height=1,
            font=("Arial", 11, "bold"),
            command=self.toggle_camera
        )
        self.btn_camera.pack(pady=(0, 4))

        # AUTO PARK button
        self.btn_autopark = tk.Button(
            master,
            text="Auto Park",
            width=10,
            height=1,
            font=("Arial", 11, "bold"),
            command=self.toggle_autopark
        )
        self.btn_autopark.pack(pady=(0, 8))

        # Status
        self.status = tk.Label(master, text="Status: STOPPED", font=("Arial", 11))
        self.status.pack(pady=(0, 8))

        # === KEYBOARD BINDINGS (hold behavior) ===
        for key in ("w", "a", "s", "d", "W", "A", "S", "D"):
            self.master.bind(f"<KeyPress-{key}>", self.on_key_press)
            self.master.bind(f"<KeyRelease-{key}>", self.on_key_release)

        # SPACE = stop
        self.master.bind("<space>", self.on_space)

        # C = toggle camera
        self.master.bind("<c>", lambda e: self.toggle_camera())
        self.master.bind("<C>", lambda e: self.toggle_camera())

        # P = toggle autopark
        self.master.bind("<p>", lambda e: self.toggle_autopark())
        self.master.bind("<P>", lambda e: self.toggle_autopark())

        # Safe close
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

        # Focus for key events
        self.master.focus_force()

    # ===== UTILITIES =====
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

    # ===== KEYBOARD MANUAL DRIVE =====
    def on_key_press(self, event):
        if self._is_autopark_running():
            # ignore manual while autopark active
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
            # space won't kill autopark; just info
            self.status.config(text="Status: AUTOPARK RUNNING (stop with Auto Park button)")
            return
        self.pressed.clear()
        self.stop()

    # ===== MOUSE (ON-SCREEN KEYS) =====
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

    # ===== MANUAL MOTION LOGIC =====
    def update_motion(self):
        # If autopark is running, ignore manual commands entirely
        if self._is_autopark_running():
            self.status.config(text="Status: AUTOPARK RUNNING (manual disabled)")
            return

        s = self._get_speed()

        if not self.pressed:
            self.stop()
            return

        forward = 'w' in self.pressed
        backward = 's' in self.pressed
        left = 'a' in self.pressed
        right = 'd' in self.pressed

        if forward and not backward:
            if left and not right:
                self.car.set_motor_model(int(0.4*s), s, s, s)
                self.status.config(text=f"Status: FORWARD-LEFT  speed={s}")
            elif right and not left:
                self.car.set_motor_model(s, s, int(0.4*s), s)
                self.status.config(text=f"Status: FORWARD-RIGHT speed={s}")
            else:
                self.car.set_motor_model(s, s, s, s)
                self.status.config(text=f"Status: FORWARD       speed={s}")

        elif backward and not forward:
            if left and not right:
                self.car.set_motor_model(int(-0.4*s), -s, -s, -s)
                self.status.config(text=f"Status: BACKWARD-LEFT  speed={s}")
            elif right and not left:
                self.car.set_motor_model(-s, -s, int(-0.4*s), -s)
                self.status.config(text=f"Status: BACKWARD-RIGHT speed={s}")
            else:
                self.car.set_motor_model(-s, -s, -s, -s)
                self.status.config(text=f"Status: BACKWARD       speed={s}")
        else:
            if left and not right:
                self.car.set_motor_model(-s, -s, s, s)
                self.status.config(text=f"Status: LEFT PIVOT      speed={s}")
            elif right and not left:
                self.car.set_motor_model(s, s, -s, -s)
                self.status.config(text=f"Status: RIGHT PIVOT     speed={s}")
            else:
                self.stop()

    # ===== BASIC ACTIONS =====
    def stop(self):
        self.car.set_motor_model(0, 0, 0, 0)
        if not self._is_autopark_running():
            self.status.config(text="Status: STOPPED")

    # ===== CAMERA CONTROL =====
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

    # ===== AUTOPARK CONTROL =====
    def toggle_autopark(self):
        if not self._is_autopark_running():
            try:
                # stop any manual motion first
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

    # ===== CLOSE HANDLING =====
    def on_close(self):
        try:
            self.pressed.clear()
            self.stop()
            self._stop_camera_process()
            self._stop_autopark_process()
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
