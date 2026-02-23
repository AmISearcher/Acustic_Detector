#!/usr/bin/env python3
import time
import signal
import threading
from collections import deque

import cv2
import numpy as np
from picamera2 import Picamera2

from base_ctrl import BaseController


# =========================
# CONFIG
# =========================
FRAME_SIZE = (640, 640)  # (W, H)

CENTER_SQUARE = 80  # pixels

GIMBAL_SERIAL_PORT = "/dev/ttyAMA0"
GIMBAL_BAUD = 115200

PAN_LIMIT = 180
TILT_LIMIT_UP = 90
TILT_LIMIT_DOWN = -30

PIX_DEADZONE = 5
PID_OUTPUT_LIMIT = 0.6

# Starting PID (will be adapted if AUTOTUNE enabled)
PAN_KP0 = 0.005
PAN_KI0 = 0.00007
PAN_KD0 = 0.0

TILT_KP0 = 0.005
TILT_KI0 = 0.00007
TILT_KD0 = 0.0

# Autotune
AUTOTUNE_DEFAULT = True
TUNE_LOG_EVERY_S = 1.0  # print PID numbers once per second


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
    def __init__(self, kp, ki, kd, output_limit, integral_limit=1e9):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_limit = float(output_limit)
        self.integral_limit = float(integral_limit)

        self.prev_error = 0.0
        self.integral = 0.0

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error: float) -> float:
        self.integral += error
        self.integral = clamp(self.integral, -self.integral_limit, self.integral_limit)

        derivative = error - self.prev_error
        self.prev_error = error

        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        return clamp(out, -self.output_limit, self.output_limit)


# =========================
# Simple Online PID Auto-Tuner (gain scheduling + oscillation damping)
# =========================
class OnlinePidTuner:
    """
    Heuristic online tuning based on recent error behavior:

    - If error is large and not improving: slowly increase Kp.
    - If error oscillates (frequent sign changes) or overshoot grows: reduce Kp, reduce Ki, optionally increase Kd a bit.
    - If error stays small but biased (steady-state): increase Ki slightly.
    """

    def __init__(
        self,
        pid: PID,
        kp_bounds=(1e-5, 0.05),
        ki_bounds=(0.0, 0.01),
        kd_bounds=(0.0, 0.2),
        window=30,              # frames history
        signflip_threshold=6,   # sign flips in window => oscillation
        improve_eps=0.85,       # "improving" if abs(err_now) < improve_eps * abs(err_old)
        large_err_px=60.0,      # consider as "large error"
        small_err_px=10.0,      # consider as "small error"
        base_rate=1.0,          # aggressiveness multiplier
    ):
        self.pid = pid
        self.kp_min, self.kp_max = kp_bounds
        self.ki_min, self.ki_max = ki_bounds
        self.kd_min, self.kd_max = kd_bounds

        self.err_hist = deque(maxlen=window)
        self.window = window

        self.signflip_threshold = signflip_threshold
        self.improve_eps = improve_eps
        self.large_err_px = large_err_px
        self.small_err_px = small_err_px

        self.base_rate = float(base_rate)

        self._last_update_t = 0.0
        self._cooldown_s = 0.15  # don't retune every single frame

    def reset(self):
        self.err_hist.clear()
        self._last_update_t = 0.0

    def set_aggressiveness(self, rate: float):
        self.base_rate = clamp(float(rate), 0.1, 5.0)

    def update(self, err: float, now: float):
        # Store
        self.err_hist.append(float(err))

        # Need enough data
        if len(self.err_hist) < max(10, self.window // 3):
            return

        # Cooldown
        if (now - self._last_update_t) < self._cooldown_s:
            return
        self._last_update_t = now

        # Metrics
        abs_err = abs(err)
        abs_old = abs(self.err_hist[0])
        abs_mid = abs(self.err_hist[len(self.err_hist)//2])

        # sign flips
        flips = 0
        prev = self.err_hist[0]
        for e in list(self.err_hist)[1:]:
            if (prev == 0.0) or (e == 0.0):
                prev = e
                continue
            if (prev > 0) != (e > 0):
                flips += 1
            prev = e

        # Overshoot-ish: did abs error grow compared to mid?
        grew_vs_mid = abs_err > (1.15 * abs_mid)

        # Improvement: compare current vs oldest in window
        improving = abs_err < (self.improve_eps * abs_old)

        # --- Tuning rules ---
        # 1) Oscillation damping
        if flips >= self.signflip_threshold or grew_vs_mid:
            # reduce Kp/Ki, add a touch of Kd (optional)
            kp = self.pid.kp * (1.0 - 0.06 * self.base_rate)
            ki = self.pid.ki * (1.0 - 0.10 * self.base_rate)
            kd = self.pid.kd + (0.0005 * self.base_rate)  # tiny bump

            self.pid.kp = clamp(kp, self.kp_min, self.kp_max)
            self.pid.ki = clamp(ki, self.ki_min, self.ki_max)
            self.pid.kd = clamp(kd, self.kd_min, self.kd_max)
            return

        # 2) Large error and not improving -> increase Kp slightly
        if abs_err >= self.large_err_px and not improving:
            kp = self.pid.kp * (1.0 + 0.05 * self.base_rate)
            self.pid.kp = clamp(kp, self.kp_min, self.kp_max)
            # keep Ki modest when far away
            self.pid.ki = clamp(self.pid.ki * (1.0 - 0.02 * self.base_rate), self.ki_min, self.ki_max)
            return

        # 3) Small error but persistent bias -> increase Ki slightly
        # bias: mean error over window not ~0
        mean_e = sum(self.err_hist) / len(self.err_hist)
        if abs_err <= self.small_err_px and abs(mean_e) > (self.small_err_px * 0.25):
            ki = self.pid.ki * (1.0 + 0.07 * self.base_rate) + 1e-7
            self.pid.ki = clamp(ki, self.ki_min, self.ki_max)
            return

        # 4) Otherwise: slowly relax Kd back to zero-ish (optional)
        if self.pid.kd > 0.0:
            self.pid.kd = clamp(self.pid.kd * (1.0 - 0.03 * self.base_rate), self.kd_min, self.kd_max)


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
    pan_pid = PID(PAN_KP0, PAN_KI0, PAN_KD0, PID_OUTPUT_LIMIT, integral_limit=20000)
    tilt_pid = PID(TILT_KP0, TILT_KI0, TILT_KD0, PID_OUTPUT_LIMIT, integral_limit=20000)

    # Online tuners
    pan_tuner = OnlinePidTuner(
        pan_pid,
        kp_bounds=(1e-5, 0.05),
        ki_bounds=(0.0, 0.01),
        kd_bounds=(0.0, 0.05),
        window=30,
        signflip_threshold=6,
        large_err_px=70.0,
        small_err_px=12.0,
        base_rate=1.0,
    )
    tilt_tuner = OnlinePidTuner(
        tilt_pid,
        kp_bounds=(1e-5, 0.05),
        ki_bounds=(0.0, 0.01),
        kd_bounds=(0.0, 0.05),
        window=30,
        signflip_threshold=6,
        large_err_px=70.0,
        small_err_px=12.0,
        base_rate=1.0,
    )

    autotune = AUTOTUNE_DEFAULT
    aggressiveness = 1.0

    tracker = None
    tracking = False

    W, H = FRAME_SIZE
    cx0 = W / 2.0
    cy0 = H / 2.0

    sq = int(CENTER_SQUARE)
    sq = max(10, min(sq, min(W, H) - 2))
    x0 = int(cx0 - sq / 2)
    y0 = int(cy0 - sq / 2)

    fps = 0.0
    last_tune_log = 0.0

    print("[INFO] Controls:")
    print("  SPACE = start tracking ROI inside the center square (CSRT init)")
    print("  r     = reset/stop tracking")
    print("  t     = toggle PID autotune")
    print("  [ / ] = decrease / increase autotune aggressiveness")
    print("  q/x   = quit")

    try:
        while not stop_event.is_set():
            t0 = time.time()

            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            now = time.time()
            mode = "idle"

            # Crosshair
            cv2.line(frame, (W // 2, 0), (W // 2, H), (0, 255, 0), 1)
            cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 0), 1)

            offx = 0.0
            offy = 0.0

            if not tracking:
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
                success, box = tracker.update(frame)
                if success:
                    mode = "tracking"
                    bx, by, bw, bh = map(int, box)

                    tx = bx + bw // 2
                    ty = by + bh // 2
                    offx = tx - cx0
                    offy = ty - cy0

                    # Autotune update (based on error only)
                    if autotune:
                        pan_tuner.set_aggressiveness(aggressiveness)
                        tilt_tuner.set_aggressiveness(aggressiveness)
                        pan_tuner.update(offx, now)
                        tilt_tuner.update(-offy, now)  # use same sign as control below

                    # PID -> gimbal
                    if abs(offx) > PIX_DEADZONE:
                        pan_angle += pan_pid.compute(offx)
                    if abs(offy) > PIX_DEADZONE:
                        # keep your original sign convention
                        tilt_angle += tilt_pid.compute(-offy)

                    pan_angle = clamp(pan_angle, -PAN_LIMIT, PAN_LIMIT)
                    tilt_angle = clamp(tilt_angle, TILT_LIMIT_DOWN, TILT_LIMIT_UP)

                    base.gimbal_ctrl(pan_angle, tilt_angle, 300, 50)

                    # UI bbox
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
                    cv2.putText(
                        frame,
                        f"Tracking  ex={offx:.0f} ey={offy:.0f}",
                        (bx, max(0, by - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )
                else:
                    tracking = False
                    tracker = None
                    pan_pid.reset()
                    tilt_pid.reset()
                    pan_tuner.reset()
                    tilt_tuner.reset()

            # FPS
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, time.time() - t0))

            # On-screen PID numbers
            y = 55
            cv2.putText(
                frame,
                f"AUTOTUNE: {'ON' if autotune else 'OFF'}  rate={aggressiveness:.2f}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            y += 22
            cv2.putText(
                frame,
                f"PAN  Kp={pan_pid.kp:.6f} Ki={pan_pid.ki:.8f} Kd={pan_pid.kd:.6f}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            y += 22
            cv2.putText(
                frame,
                f"TILT Kp={tilt_pid.kp:.6f} Ki={tilt_pid.ki:.8f} Kd={tilt_pid.kd:.6f}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            # Footer
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

            # Console logs (once per second)
            if (now - last_tune_log) >= TUNE_LOG_EVERY_S:
                last_tune_log = now
                print(
                    f"[PID] autotune={'ON' if autotune else 'OFF'} rate={aggressiveness:.2f} "
                    f"PAN(kp={pan_pid.kp:.6f}, ki={pan_pid.ki:.8f}, kd={pan_pid.kd:.6f}) "
                    f"TILT(kp={tilt_pid.kp:.6f}, ki={tilt_pid.ki:.8f}, kd={tilt_pid.kd:.6f}) "
                    f"err=({offx:.1f},{offy:.1f})"
                )

            cv2.imshow("Tracker Only + Online PID Autotune", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("x")):
                stop_event.set()

            elif key == ord("r"):
                tracking = False
                tracker = None
                pan_pid.reset()
                tilt_pid.reset()
                pan_tuner.reset()
                tilt_tuner.reset()

            elif key == ord("t"):
                autotune = not autotune

            elif key == ord("["):
                aggressiveness = clamp(aggressiveness * 0.85, 0.1, 5.0)

            elif key == ord("]"):
                aggressiveness = clamp(aggressiveness * 1.15, 0.1, 5.0)

            elif key == ord(" "):
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x0, y0, sq, sq))
                tracking = True
                pan_pid.reset()
                tilt_pid.reset()
                pan_tuner.reset()
                tilt_tuner.reset()

    finally:
        stop_event.set()
        try:
            picam2.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

        # Re-send last pose briefly (optional)
        try:
            for _ in range(5):
                base.gimbal_ctrl(pan_angle, tilt_angle, 300, 50)
                time.sleep(0.03)
        except Exception:
            pass

        print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()
