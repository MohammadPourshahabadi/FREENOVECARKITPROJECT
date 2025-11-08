#!/usr/bin/env python3
import tkinter as tk
import sys

from motor import Ordinary_Car   # Uses Freenove's driver (FNK0043 kit compatible)

# PWM limits for this kit (Freenove clamps to [-4095, 4095])
SPEED_MIN = 0
SPEED_MAX = 4095
DEFAULT_SPEED = 2000  # Safe starting point

class ManualDashboard:
    def __init__(self, master):
        self.master = master
        self.master.title("Freenove 4WD - Manual Dashboard")

        # Init car controller
        self.car = Ordinary_Car()

        # Speed variable (bound to slider)
        self.speed = tk.IntVar(value=DEFAULT_SPEED)

        # === UI LAYOUT ===
        title = tk.Label(
            master,
            text="Freenove 4WD Manual Control",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=8)

        hint = tk.Label(
            master,
            text="Buttons or keys: W/A/S/D, SPACE to stop",
            font=("Arial", 10)
        )
        hint.pack(pady=(0, 8))

        # Speed slider
        speed_frame = tk.Frame(master)
        speed_frame.pack(pady=5)
        tk.Label(speed_frame, text="Speed:", font=("Arial", 11)).pack(side=tk.LEFT, padx=(0, 5))
        self.speed_scale = tk.Scale(
            speed_frame,
            from_=SPEED_MIN,
            to=SPEED_MAX,
            orient=tk.HORIZONTAL,
            variable=self.speed,
            length=260
        )
        self.speed_scale.pack(side=tk.LEFT)

        # Control buttons grid
        btn_frame = tk.Frame(master)
        btn_frame.pack(pady=10)

        self.btn_forward = tk.Button(
            btn_frame, text="▲ Forward",
            width=12, height=2,
            command=self.forward
        )
        self.btn_forward.grid(row=0, column=1, padx=5, pady=5)

        self.btn_left = tk.Button(
            btn_frame, text="◀ Left",
            width=12, height=2,
            command=self.left
        )
        self.btn_left.grid(row=1, column=0, padx=5, pady=5)

        self.btn_stop = tk.Button(
            btn_frame, text="■ STOP",
            width=12, height=2,
            command=self.stop
        )
        self.btn_stop.grid(row=1, column=1, padx=5, pady=5)

        self.btn_right = tk.Button(
            btn_frame, text="Right ▶",
            width=12, height=2,
            command=self.right
        )
        self.btn_right.grid(row=1, column=2, padx=5, pady=5)

        self.btn_backward = tk.Button(
            btn_frame, text="▼ Backward",
            width=12, height=2,
            command=self.backward
        )
        self.btn_backward.grid(row=2, column=1, padx=5, pady=5)

        # Status label
        self.status = tk.Label(master, text="Status: STOPPED", font=("Arial", 11))
        self.status.pack(pady=5)

        # Key bindings
        self.master.bind("<w>", lambda e: self.forward())
        self.master.bind("<W>", lambda e: self.forward())
        self.master.bind("<s>", lambda e: self.backward())
        self.master.bind("<S>", lambda e: self.backward())
        self.master.bind("<a>", lambda e: self.left())
        self.master.bind("<A>", lambda e: self.left())
        self.master.bind("<d>", lambda e: self.right())
        self.master.bind("<D>", lambda e: self.right())
        self.master.bind("<space>", lambda e: self.stop())

        # Safe close
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    # === MOTION HELPERS ===

    def _get_speed(self):
        s = self.speed.get()
        if s < SPEED_MIN:
            s = SPEED_MIN
        if s > SPEED_MAX:
            s = SPEED_MAX
        return s

    def forward(self):
        s = self._get_speed()
        self.car.set_motor_model(s, s, s, s)
        self.status.config(text=f"Status: FORWARD (speed={s})")

    def backward(self):
        s = self._get_speed()
        self.car.set_motor_model(-s, -s, -s, -s)
        self.status.config(text=f"Status: BACKWARD (speed={s})")

    def left(self):
        s = self._get_speed()
        # Left wheels slower/backward, right wheels forward -> pivot left
        self.car.set_motor_model(-s, -s, s, s)
        self.status.config(text=f"Status: LEFT TURN (speed={s})")

    def right(self):
        s = self._get_speed()
        # Right wheels slower/backward, left wheels forward -> pivot right
        self.car.set_motor_model(s, s, -s, -s)
        self.status.config(text=f"Status: RIGHT TURN (speed={s})")

    def stop(self):
        self.car.set_motor_model(0, 0, 0, 0)
        self.status.config(text="Status: STOPPED")

    def on_close(self):
        # Safety: always stop & release on exit
        try:
            self.stop()
            if hasattr(self.car, "close"):
                self.car.close()
        except Exception as e:
            print("Error during close:", e)
        self.master.destroy()


def main():
    root = tk.Tk()
    app = ManualDashboard(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        # If closed from terminal
        app.on_close()
        sys.exit(0)


if __name__ == "__main__":
    main()
