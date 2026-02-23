#!/usr/bin/env python3
import time
import signal
import threading

import cv2
import numpy as np
from picamera2 import Picamera2

from base_ctrl import BaseController


# =========================
# CONFIG
# =========================
# Camera preview size (must match Picamera2 config)
FRAME_SIZE = (640, 640)  # (W, H)

# Center "capture" square size (pixels)
CENTER_SQUARE = 80  # change to 60/100/etc

# Gimbal serial
GIMBAL_SERIAL_PORT = "/dev/ttyAMA0"
GIMBAL_BAUD = 115200

# Servo mechanical limits
PAN_LIMIT = 180
TILT_LIMIT_UP = 90
TILT_LIMIT_DOWN = -30

# PID / deadzone
PIX_DEADZONE = 5
PAN_KP = 0.005
TILT_KP = 0.005
PID_OUTPUT_LIMIT = 0.6


# =========================
# STOP CONTROL
# =========================
stop_event = threading.Event()

def _handle_sig(*_):
    stop_event.set()

signal.signal(signal.SIGINT, _handle_sig)
signal.signal(signal.SIGTERM, _handle_sig)


# =========================
# UTILS
# =========================
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# =========================
# PID
# =========================
class PID:
    def __init__(self, kp, ki, kd, output_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.prev_error = 0.0
        self.integral = 0.0

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        return clamp(out, -self.output_limit, self.output_limit)


# =========================
# MAIN
# =========================
def main():
    # Camera
    picam2 = Picamera2(1)
    picam2.configure(
        picam2.create_preview_configuration(main={"format": "RGB888", "size": FRAME_SIZE})
    )
    picam2.start()

    # Gimbal
    base = BaseController(GIMBAL_SERIAL_PORT, GIMBAL_BAUD)
    pan_angle, tilt_angle = 0.0, 0.0
    base.gimbal_ctrl(pan_angle, tilt_angle, 300, 50)

    # PID
    pan_pid = PID(kp=PAN_KP, ki=0.00007, kd=0.0, output_limit=PID_OUTPUT_LIMIT)
    tilt_pid = PID(kp=TILT_KP, ki=0.00007, kd=0.0, output_limit=PID_OUTPUT_LIMIT)

    tracker = None
    tracking = False

    # Frame center
    W, H = FRAME_SIZE
    cx0 = W / 2.0
    cy0 = H / 2.0

    # Precompute center square bbox (x, y, w, h)
    sq = int(CENTER_SQUARE)
    sq = max(10, min(sq, min(W, H) - 2))
    x0 = int(cx0 - sq / 2)
    y0 = int(cy0 - sq / 2)

    # FPS
    fps = 0.0

    print("[INFO] Controls:")
    print("  SPACE = start tracking the object currently inside the center square")
    print("  r     = reset/stop tracking")
    print("  q/x   = quit")

    try:
        while not stop_event.is_set():
            t0 = time.time()

            # Capture RGB -> BGR
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            mode = "idle"

            # Draw crosshair
            cv2.line(frame, (W // 2, 0), (W // 2, H), (0, 255, 0), 1)
            cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 0), 1)

            if not tracking:
                # Draw the center square guide
                cv2.rectangle(frame, (x0, y0), (x0 + sq, y0 + sq), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    "Press SPACE to lock & track",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
            else:
                # Update tracker
                success, box = tracker.update(frame)
                if success:
                    mode = "tracking"
                    bx, by, bw, bh = map(int, box)

                    tx = bx + bw // 2
                    ty = by + bh // 2
                    offx = tx - cx0
                    offy = ty - cy0

                    # PID -> gimbal
                    if abs(offx) > PIX_DEADZONE:
                        pan_angle += pan_pid.compute(offx)
                    if abs(offy) > PIX_DEADZONE:
                        # keep your original sign convention from your script
                        tilt_angle += tilt_pid.compute(-offy)

                    pan_angle = clamp(pan_angle, -PAN_LIMIT, PAN_LIMIT)
                    tilt_angle = clamp(tilt_angle, TILT_LIMIT_DOWN, TILT_LIMIT_UP)

                    base.gimbal_ctrl(pan_angle, tilt_angle, 300, 50)

                    # UI bbox
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
                    cv2.putText(
                        frame,
                        "Tracking",
                        (bx, max(0, by - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )
                else:
                    # Lost
                    tracking = False
                    tracker = None
                    pan_pid.reset()
                    tilt_pid.reset()

            # FPS + status
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, time.time() - t0))
            cv2.putText(
                frame,
                f"Mode: {mode}",
                (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (W - 140, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Tracker Only (SPACE to start)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("x")):
                stop_event.set()

            elif key == ord("r"):
                tracking = False
                tracker = None
                pan_pid.reset()
                tilt_pid.reset()

            elif key == ord(" "):
                # Start tracking using the center square bbox
                # (we "capture" what's inside the square by initializing tracker on that ROI)
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x0, y0, sq, sq))
                tracking = True
                pan_pid.reset()
                tilt_pid.reset()

    finally:
        stop_event.set()
        try:
            picam2.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

        # optional: re-send last pose a few times to avoid twitch (same idea as your big script)
        try:
            for _ in range(5):
                base.gimbal_ctrl(pan_angle, tilt_angle, 300, 50)
                time.sleep(0.03)
        except Exception:
            pass

        print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()
