#!/usr/bin/env python3
import tkinter as tk
import sys
from motor import Ordinary_Car

SPEED_MIN = 0
SPEED_MAX = 4095
DEFAULT_SPEED = 1100  # ← your default speed

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

        # Which keys are currently held down
        self.pressed = set()

        # Speed control
        self.speed = tk.IntVar(value=DEFAULT_SPEED)

        # === UI ===
        title = tk.Label(
            master,
            text="Freenove 4WD Manual Control (Hold to Move)",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=(8, 2))

        subtitle = tk.Label(
            master,
            text="Keyboard: hold W/A/S/D  |  SPACE = Stop\nMouse: press key = Move, release = Stop",
            font=("Arial", 10)
        )
        subtitle.pack(pady=(0, 8))

        # Speed slider
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

        # On-screen "keyboard"
        kb_frame = tk.Frame(master)
        kb_frame.pack(pady=10)

        def make_key(text, row, col, press_cmd, release_cmd, w=6, bg=KEY_BG, abg=KEY_ACTIVE_BG):
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
        self.btn_w = make_key("W", 0, 1, self._press_forward, self._release_dir)
        self.btn_a = make_key("A", 1, 0, self._press_left,    self._release_dir)
        self.btn_s = make_key("S", 1, 1, self._press_backward,self._release_dir)
        self.btn_d = make_key("D", 1, 2, self._press_right,   self._release_dir)

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
        self.btn_stop.pack(pady=(0, 10))

        # Status
        self.status = tk.Label(master, text="Status: STOPPED", font=("Arial", 11))
        self.status.pack(pady=(0, 8))

        # Keyboard bindings
        for key in ("w", "a", "s", "d", "W", "A", "S", "D"):
            self.master.bind(f"<KeyPress-{key}>", self.on_key_press)
            self.master.bind(f"<KeyRelease-{key}>", self.on_key_release)

        self.master.bind("<space>", self.on_space)

        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        self.master.focus_force()

    def _get_speed(self):
        s = self.speed.get()
        if s < SPEED_MIN:
            s = SPEED_MIN
        if s > SPEED_MAX:
            s = SPEED_MAX
        return s

    # --- Keyboard handlers ---
    def on_key_press(self, event):
        key = event.keysym.lower()
        if key in ("w", "a", "s", "d"):
            if key not in self.pressed:
                self.pressed.add(key)
                self.update_motion()

    def on_key_release(self, event):
        key = event.keysym.lower()
        if key in self.pressed:
            self.pressed.remove(key)
            self.update_motion()

    def on_space(self, event):
        self.pressed.clear()
        self.stop()

    # --- Mouse press helpers ---
    def _press_forward(self):
        self.pressed = {"w"}
        self.update_motion()

    def _press_backward(self):
        self.pressed = {"s"}
        self.update_motion()

    def _press_left(self):
        self.pressed = {"a"}
        self.update_motion()

    def _press_right(self):
        self.pressed = {"d"}
        self.update_motion()

    def _release_dir(self):
        self.pressed.clear()
        self.update_motion()

    # --- Motion logic ---
    def update_motion(self):
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

    def stop(self):
        self.car.set_motor_model(0, 0, 0, 0)
        self.status.config(text="Status: STOPPED")

    def on_close(self):
        try:
            self.stop()
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
