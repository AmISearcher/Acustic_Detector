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
FRAME_SIZE = (640, 640)        # (W, H)
CENTER_SQUARE = 80             # pixels

# NEW PTZ over USB (Virtual COM Port)
GIMBAL_USB_PORT = "/dev/ttyUSB0"
GIMBAL_BAUD = 115200
GIMBAL_INIT_SLEEP_S = 2.0      # wait MCU reset after opening port

# Servo mechanical limits (absolute degrees)
PAN_LIMIT = 180
TILT_LIMIT_UP = 90
TILT_LIMIT_DOWN = -30

# PID / deadzone
PIX_DEADZONE = 5
PAN_KP = 0.005
TILT_KP = 0.005
PID_OUTPUT_LIMIT = 0.6

# PTZ behavior
SET_ZERO_ON_START = True       # sends "Z\n" at start to define (0,0)
QUERY_POS_EVERY_S = 0.0        # 0 disables. If >0 -> send "?\n" periodically and print/overlay.


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
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_limit = float(output_limit)
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
# NEW PTZ USB DRIVER (per your doc)
# =========================
class UsbPTZ:
    """
    Implements:
      I\\n -> DEVICE:...
      Z\\n -> OK:ZERO
      P<deg>\\n
      T<deg>\\n
      S\\n
      ?\\n -> POS:pan,tilt
    """
    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.1):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(GIMBAL_INIT_SLEEP_S)  # wait for MCU reset
        self.last_pan = 0.0
        self.last_tilt = 0.0

    def _write_line(self, s: str):
        # Commands must end with \n (LF)
        self.ser.write((s + "\n").encode("ascii", errors="ignore"))

    def identify(self) -> str:
        self._write_line("I")
        return self._read_line_best_effort()

    def set_zero(self):
        self._write_line("Z")
        # optional read
        _ = self._read_line_best_effort()

    def stop(self):
        self._write_line("S")
        _ = self._read_line_best_effort()

    def move_abs(self, pan_deg: float, tilt_deg: float):
        # Absolute degrees
        self.last_pan = float(pan_deg)
        self.last_tilt = float(tilt_deg)
        self._write_line(f"P{pan_deg:.2f}")
        self._write_line(f"T{tilt_deg:.2f}")

    def query_pos(self):
        self._write_line("?")
        line = self._read_line_best_effort()
        # Expected: POS:90.50,-15.00
        if not line:
            return None
        line = line.strip()
        if line.startswith("POS:"):
            try:
                payload = line.split("POS:", 1)[1]
                a, b = payload.split(",", 1)
                return float(a), float(b)
            except Exception:
                return None
        return None

    def _read_line_best_effort(self) -> str:
        try:
            raw = self.ser.readline()
            if not raw:
                return ""
            return raw.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


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

    # PTZ over USB
    ptz = UsbPTZ(GIMBAL_USB_PORT, GIMBAL_BAUD, timeout=0.1)
    id_line = ptz.identify()
    print(f"[INFO] PTZ ID: {id_line}")

    if SET_ZERO_ON_START:
        ptz.set_zero()
        print("[INFO] PTZ ZERO set (Z).")

    # Current absolute angles (controller expects absolute)
    pan_angle, tilt_angle = 0.0, 0.0
    ptz.move_abs(pan_angle, tilt_angle)

    # PID
    pan_pid = PID(kp=PAN_KP, ki=0.00007, kd=0.0, output_limit=PID_OUTPUT_LIMIT)
    tilt_pid = PID(kp=TILT_KP, ki=0.00007, kd=0.0, output_limit=PID_OUTPUT_LIMIT)

    tracker = None
    tracking = False

    # Frame center
    W, H = FRAME_SIZE
    cx0 = W / 2.0
    cy0 = H / 2.0

    # Center square ROI
    sq = int(CENTER_SQUARE)
    sq = max(10, min(sq, min(W, H) - 2))
    x0 = int(cx0 - sq / 2)
    y0 = int(cy0 - sq / 2)

    fps = 0.0

    last_pos_query = 0.0
    last_pos = None

    print("[INFO] Controls:")
    print("  SPACE = start tracking ROI inside the center square (CSRT init)")
    print("  r     = reset/stop tracking")
    print("  q/x   = quit")

    try:
        while not stop_event.is_set():
            t0 = time.time()

            # Capture
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            now = time.time()
            mode = "idle"

            # Crosshair
            cv2.line(frame, (W // 2, 0), (W // 2, H), (0, 255, 0), 1)
            cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 0), 1)

            # Optional PTZ position query
            if QUERY_POS_EVERY_S > 0 and (now - last_pos_query) >= QUERY_POS_EVERY_S:
                last_pos_query = now
                last_pos = ptz.query_pos()

            if not tracking:
                # Draw ROI square guide
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

                    # PID -> absolute angles update
                    if abs(offx) > PIX_DEADZONE:
                        pan_angle += pan_pid.compute(offx)

                    if abs(offy) > PIX_DEADZONE:
                        # keep your sign convention: image Y grows down => invert
                        tilt_angle += tilt_pid.compute(-offy)

                    pan_angle = clamp(pan_angle, -PAN_LIMIT, PAN_LIMIT)
                    tilt_angle = clamp(tilt_angle, TILT_LIMIT_DOWN, TILT_LIMIT_UP)

                    # Send absolute coords to PTZ controller (no speed params per doc)
                    ptz.move_abs(pan_angle, tilt_angle)

                    # UI bbox
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
                    cv2.putText(
                        frame,
                        f"Tracking ex={offx:.0f} ey={offy:.0f}",
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

            # show angles (absolute)
            y = 55
            cv2.putText(
                frame,
                f"PTZ cmd: pan={pan_angle:.2f} tilt={tilt_angle:.2f}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            y += 22
            if last_pos is not None:
                cv2.putText(
                    frame,
                    f"PTZ pos: pan={last_pos[0]:.2f} tilt={last_pos[1]:.2f}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Tracker Only (USB PTZ)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("x")):
                stop_event.set()

            elif key == ord("r"):
                tracking = False
                tracker = None
                pan_pid.reset()
                tilt_pid.reset()

            elif key == ord(" "):
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x0, y0, sq, sq))
                tracking = True
                pan_pid.reset()
                tilt_pid.reset()

    finally:
        stop_event.set()

        # try to STOP motors (emergency stop per doc)
        try:
            ptz.stop()
        except Exception:
            pass

        try:
            picam2.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

        try:
            ptz.close()
        except Exception:
            pass

        print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()
