#!/usr/bin/env python3
import time
import signal
import threading

import cv2
from picamera2 import Picamera2
import serial


# =========================
# CONFIG
# =========================
FRAME_SIZE = (640, 640)
CENTER_SQUARE = 80

GIMBAL_USB_PORT = "/dev/ttyUSB0"
GIMBAL_BAUD = 115200
GIMBAL_INIT_SLEEP_S = 2.0

# Firmware limits (real clamp may be inside MCU)
PAN_LIMIT = 2880
TILT_LIMIT_UP = 900
TILT_LIMIT_DOWN = -900

# Tracking tuning
PIX_DEADZONE = 3

# PID now outputs SPEED (deg/s), not position step
PAN_KP = 2.0
TILT_KP = 2.0

MAX_PAN_SPEED = 570.0
MAX_PAN_ACCEL = 1500.0

MAX_TILT_SPEED = 220.0        # менша макс. швидкість по tilt
MAX_TILT_ACCEL = 350.0        # набагато плавніше прискорення
MAX_TILT_DECEL = 700.0        # гальмування можна швидше ніж розгін

PTZ_SEND_HZ = 60.0
MIN_DEG_STEP = 0.5

SET_ZERO_ON_START = True


# =========================
# STOP CONTROL
# =========================
stop_event = threading.Event()
signal.signal(signal.SIGINT, lambda *_: stop_event.set())
signal.signal(signal.SIGTERM, lambda *_: stop_event.set())


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# =========================
# USB PTZ
# =========================
class UsbPTZ:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(GIMBAL_INIT_SLEEP_S)

    def send(self, cmd):
        self.ser.write((cmd + "\n").encode("ascii"))

    def identify(self):
        self.send("I")
        return self.ser.readline().decode(errors="replace").strip()

    def set_zero(self):
        self.send("Z")

    def stop(self):
        self.send("S")

    def move_abs(self, pan, tilt):
        self.send(f"P{pan:.2f}")
        self.send(f"T{tilt:.2f}")

    def close(self):
        self.ser.close()

def accel_decel(current, target, accel_up, accel_down, dt):
    """
    accel_up   = max accel when speeding up in same direction (deg/s^2)
    accel_down = max decel when slowing down / reversing (deg/s^2)
    """
    diff = target - current

    # if we're reducing magnitude or reversing direction -> use decel limit
    same_dir = (current == 0.0) or (target == 0.0) or ((current > 0) == (target > 0))
    reducing_mag = abs(target) < abs(current)

    use_decel = (not same_dir) or reducing_mag
    a = accel_down if use_decel else accel_up

    max_step = a * dt
    if diff > max_step:
        diff = max_step
    elif diff < -max_step:
        diff = -max_step

    return current + diff
# =========================
# MAIN
# =========================
def main():

    picam2 = Picamera2(1)
    picam2.configure(
        picam2.create_preview_configuration(main={"format": "RGB888", "size": FRAME_SIZE})
    )
    picam2.start()

    ptz = UsbPTZ(GIMBAL_USB_PORT, GIMBAL_BAUD)
    print("[INFO] PTZ ID:", ptz.identify())

    if SET_ZERO_ON_START:
        ptz.set_zero()

    pan = 0.0
    tilt = 0.0

    current_pan_speed = 0.0
    current_tilt_speed = 0.0

    last_send = 0.0
    send_dt = 1.0 / PTZ_SEND_HZ

    last_sent_pan = pan
    last_sent_tilt = tilt

    tracker = None
    tracking = False

    W, H = FRAME_SIZE
    cx0 = W / 2.0
    cy0 = H / 2.0

    sq = CENTER_SQUARE
    x0 = int(cx0 - sq / 2)
    y0 = int(cy0 - sq / 2)

    t_prev = time.time()

    try:
        while not stop_event.is_set():

            now = time.time()
            dt = now - t_prev
            t_prev = now
            dt = clamp(dt, 0.0, 0.05)  # max 50ms step
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if not tracking:
                cv2.rectangle(frame, (x0, y0), (x0 + sq, y0 + sq), (0,255,255), 2)
            else:
                ok, box = tracker.update(frame)
                if ok:
                    bx, by, bw, bh = map(int, box)
                    tx = bx + bw // 2
                    ty = by + bh // 2

                    offx = tx - cx0
                    offy = ty - cy0

                    # PID -> target speed
                    target_pan_speed = clamp(PAN_KP * offx, -MAX_PAN_SPEED, MAX_PAN_SPEED)
                    target_tilt_speed = clamp(-TILT_KP * offy, -MAX_TILT_SPEED, MAX_TILT_SPEED)
                    # If direction flips hard, force a brief zero-crossing to avoid "stuck" behavior
                    if current_tilt_speed != 0.0 and target_tilt_speed != 0.0:
                        if (current_tilt_speed > 0) != (target_tilt_speed > 0):
                            # require passing through near-zero before reversing hard
                            if abs(current_tilt_speed) > 40.0:   # threshold deg/s
                                target_tilt_speed = 0.0

                    if abs(offx) < PIX_DEADZONE:
                        target_pan_speed = 0.0
                    if abs(offy) < PIX_DEADZONE:
                        target_tilt_speed = 0.0

                    # Acceleration limiting
                    def accel(current, target, max_accel):
                        diff = target - current
                        max_step = max_accel * dt
                        diff = clamp(diff, -max_step, max_step)
                        return current + diff

                    current_pan_speed  = accel_decel(current_pan_speed,  target_pan_speed,
                                                     MAX_PAN_ACCEL, MAX_PAN_ACCEL*1.2, dt)

                    current_tilt_speed = accel_decel(current_tilt_speed, target_tilt_speed,
                                                     MAX_TILT_ACCEL, MAX_TILT_DECEL, dt)

                    # Integrate velocity -> position
                    pan += current_pan_speed * dt
                    tilt += current_tilt_speed * dt

                    pan = clamp(pan, -PAN_LIMIT, PAN_LIMIT)
                    tilt = clamp(tilt, TILT_LIMIT_DOWN, TILT_LIMIT_UP)

                    # if we hit limits, kill speed pushing into the limit
                    if (tilt >= TILT_LIMIT_UP and current_tilt_speed > 0):
                        current_tilt_speed = 0.0
                    if (tilt <= TILT_LIMIT_DOWN and current_tilt_speed < 0):
                        current_tilt_speed = 0.0

                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0,255,255), 2)

                else:
                    tracking = False
                    tracker = None
                    current_pan_speed = 0.0
                    current_tilt_speed = 0.0
                    ptz.stop()

            # Rate-limited send (send when PAN or TILT changes enough)
            if (now - last_send) >= send_dt:
                pan_changed = abs(pan - last_sent_pan) >= MIN_DEG_STEP
                tilt_changed = abs(tilt - last_sent_tilt) >= (MIN_DEG_STEP * 0.5)  # smaller step for tilt

                if pan_changed or tilt_changed:
                    ptz.move_abs(pan, tilt)
                    last_sent_pan = pan
                    last_sent_tilt = tilt

                last_send = now

            cv2.imshow("FAST Tracking PTZ", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('x')):
                break
            elif key == ord(' '):
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x0, y0, sq, sq))
                tracking = True
            elif key == ord('r'):
                tracking = False
                tracker = None
                current_pan_speed = 0.0
                current_tilt_speed = 0.0
            elif key == ord('s'):
                current_pan_speed = 0.0
                current_tilt_speed = 0.0
                ptz.stop()

    finally:
        ptz.stop()
        ptz.close()
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
