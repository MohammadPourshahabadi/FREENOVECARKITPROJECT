#!/usr/bin/env python3
import tkinter as tk
import sys
from motor import Ordinary_Car

SPEED_MIN = 0
SPEED_MAX = 4095
DEFAULT_SPEED = 1800  # adjust if too fast/slow

KEY_BG = "#eeeeee"
KEY_ACTIVE_BG = "#cccccc"
STOP_BG = "#ff6666"
STOP_ACTIVE_BG = "#ff3333"

class ManualKeyDashboard:
    def __init__(self, master):
        self.master = master
        self.master.title("Freenove 4WD - Manual Key Dashboard")

        # Init car
        self.car = Ordinary_Car()

        # Speed control
        self.speed = tk.IntVar(value=DEFAULT_SPEED)

        title = tk.Label(
            master,
            text="Freenove 4WD Manual Control",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=(8, 2))

        subtitle = tk.Label(
            master,
            text="Use on-screen keys (click) or keyboard: W/A/S/D, SPACE = Stop",
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

        # "Keyboard" layout frame
        kb_frame = tk.Frame(master)
        kb_frame.pack(pady=10)

        # Helper to create a key-like button
        def make_key(text, row, col, cmd, w=6):
            btn = tk.Button(
                kb_frame,
                text=text,
                width=w,
                height=2,
                font=("Arial", 14, "bold"),
                bg=KEY_BG,
                activebackground=KEY_ACTIVE_BG,
                command=cmd
            )
            btn.grid(row=row, column=col, padx=5, pady=5)
            return btn

        # Layout:
        #    [   ][ W ][   ]
        #    [ A ][ S ][ D ]
        # STOP button below

        self.btn_w = make_key("W", 0, 1, self.forward)
        self.btn_a = make_key("A", 1, 0, self.left)
        self.btn_s = make_key("S", 1, 1, self.backward)
        self.btn_d = make_key("D", 1, 2, self.right)

        # STOP key (big red)
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

        # Status label
        self.status = tk.Label(master, text="Status: STOPPED", font=("Arial", 11))
        self.status.pack(pady=(0, 8))

        # Bind keyboard keys
        self.master.bind("<w>", lambda e: self.forward())
        self.master.bind("<W>", lambda e: self.forward())
        self.master.bind("<a>", lambda e: self.left())
        self.master.bind("<A>", lambda e: self.left())
        self.master.bind("<s>", lambda e: self.backward())
        self.master.bind("<S>", lambda e: self.backward())
        self.master.bind("<d>", lambda e: self.right())
        self.master.bind("<D>", lambda e: self.right())
        self.master.bind("<space>", lambda e: self.stop())

        # Safe close
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---- Helpers ----
    def _get_speed(self):
        s = self.speed.get()
        if s < SPEED_MIN:
            s = SPEED_MIN
        if s > SPEED_MAX:
            s = SPEED_MAX
        return s

    # ---- Movements (W/A/S/D) ----
    def forward(self):
        s = self._get_speed()
        self.car.set_motor_model(s, s, s, s)
        self.status.config(text=f"Status: FORWARD (W)  speed={s}")

    def backward(self):
        s = self._get_speed()
        self.car.set_motor_model(-s, -s, -s, -s)
        self.status.config(text=f"Status: BACKWARD (S) speed={s}")

    def left(self):
        s = self._get_speed()
        # Pivot-style left: left wheels backward, right wheels forward
        self.car.set_motor_model(-s, -s, s, s)
        self.status.config(text=f"Status: LEFT (A)     speed={s}")

    def right(self):
        s = self._get_speed()
        # Pivot-style right: left wheels forward, right wheels backward
        self.car.set_motor_model(s, s, -s, -s)
        self.status.config(text=f"Status: RIGHT (D)    speed={s}")

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
    app = ManualKeyDashboard(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()
        sys.exit(0)

if __name__ == "__main__":
    main()
