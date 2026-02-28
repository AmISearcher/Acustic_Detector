#!/usr/bin/env python3
import time
import csv
import threading
import signal

import cv2
import numpy as np

import degirum as dg
from picamera2 import Picamera2
import serial

from laser_reader_thread import start_laser_reader, get_laser_distance


# =========================
# CONFIG
# =========================
# Visual
FRAME_SIZE = (640, 640)
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
REDETECT_INTERVAL = 0.5   # seconds

# PTZ (USB firmware)
GIMBAL_USB_PORT = "/dev/ttyUSB0"
GIMBAL_BAUD = 115200
GIMBAL_INIT_SLEEP_S = 2.0
SET_ZERO_ON_START = True

# PTZ limits (your fast PTZ script)
PAN_LIMIT = 2880.0
TILT_LIMIT_UP = 900.0
TILT_LIMIT_DOWN = -900.0

# Tracking tuning (speed-based)
PIX_DEADZONE = 5
PAN_KP = 2.0
TILT_KP = 2.0

MAX_PAN_SPEED = 570.0
MAX_PAN_ACCEL = 1500.0

MAX_TILT_SPEED = 220.0
MAX_TILT_ACCEL = 350.0
MAX_TILT_DECEL = 700.0

PTZ_SEND_HZ = 60.0
MIN_DEG_STEP = 0.5

# Manual aiming (works only when NOT tracking)
MANUAL_PAN_STEP = 40.0
MANUAL_TILT_STEP = 20.0

# Keys:
#  - WASD: move
#  - z: zero
#  - k: stop
#  - space: force immediate detection next loop
#  - r: reset tracking
#  - q/x: quit


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
# PTZ USB API
# =========================
class UsbPTZ:
    def __init__(self, port: str, baud: int):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(GIMBAL_INIT_SLEEP_S)

    def send(self, cmd: str):
        self.ser.write((cmd + "\n").encode("ascii"))

    def identify(self) -> str:
        self.send("I")
        return self.ser.readline().decode(errors="replace").strip()

    def set_zero(self):
        self.send("Z")

    def stop(self):
        self.send("S")

    def move_abs(self, pan: float, tilt: float):
        self.send(f"P{pan:.2f}")
        self.send(f"T{tilt:.2f}")

    def close(self):
        self.ser.close()


def accel_decel(current: float, target: float, accel_up: float, accel_down: float, dt: float) -> float:
    diff = target - current

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
# Detection helpers
# =========================
def apply_nms_xyxy(detections, iou_thresh=0.5):
    if not detections:
        return []
    boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
    scores = np.array([d["score"] for d in detections], dtype=np.float32)

    xywh = boxes.copy()
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]

    idx = cv2.dnn.NMSBoxes(
        bboxes=xywh.tolist(),
        scores=scores.tolist(),
        score_threshold=CONF_THRESHOLD,
        nms_threshold=iou_thresh,
    )
    if len(idx) == 0:
        return []
    if isinstance(idx, np.ndarray):
        idx = idx.flatten().tolist()
    elif isinstance(idx, list) and isinstance(idx[0], (list, tuple)):
        idx = [i[0] for i in idx]
    return [detections[i] for i in idx]


def run_detection(model, frame_bgr):
    res = model.predict(frame_bgr)
    dets = []
    for det in res.results:
        if det.get("score", 0.0) >= CONF_THRESHOLD:
            dets.append(det)
    return apply_nms_xyxy(dets, IOU_THRESHOLD)


# =========================
# MAIN
# =========================
def main():
    # Start laser thread
    start_laser_reader()

    # Load model
    model = dg.load_model(
        model_name="best",
        inference_host_address="@local",
        zoo_url="../models/models_640_resized_hq",
        token="",
        overlay_color=(0, 255, 0),
    )

    # Camera
    picam2 = Picamera2(0)
    picam2.configure(
        picam2.create_preview_configuration(main={"format": "RGB888", "size": FRAME_SIZE})
    )
    picam2.start()

    # PTZ
    ptz = UsbPTZ(GIMBAL_USB_PORT, GIMBAL_BAUD)
    print("[INFO] PTZ ID:", ptz.identify())

    # Set zero on start (and sync software state)
    pan = 0.0
    tilt = 0.0
    if SET_ZERO_ON_START:
        ptz.set_zero()
        time.sleep(0.2)
        pan, tilt = 0.0, 0.0
        ptz.move_abs(pan, tilt)

    # Speed state for tracking
    current_pan_speed = 0.0
    current_tilt_speed = 0.0

    # Rate-limited send
    last_send = 0.0
    send_dt = 1.0 / PTZ_SEND_HZ
    last_sent_pan = pan
    last_sent_tilt = tilt

    # Tracking state
    tracker = None
    tracking = False
    last_detection_time = 0.0

    # Frame center
    cx0 = FRAME_SIZE[0] / 2.0
    cy0 = FRAME_SIZE[1] / 2.0

    # Logging
    log_file = open("tracking_log.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "timestamp",
        "mode",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "target_cx", "target_cy",
        "offset_x", "offset_y",
        "pan", "tilt",
        "pan_speed", "tilt_speed",
        "laser_m",
        "fps",
    ])

    fps = 0.0
    t_prev = time.time()

    # helper: force detection
    force_detect = False

    try:
        while not stop_event.is_set():
            t0 = time.time()
            now = time.time()

            # dt for speed integration
            dt = now - t_prev
            t_prev = now
            dt = clamp(dt, 0.0, 0.05)

            # Capture
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            mode = "idle"
            bbox_x = bbox_y = bbox_w = bbox_h = -1
            tx = ty = -1
            offx = offy = 0

            # =========================
            # DETECT periodically (same style)
            # =========================
            if force_detect or ((now - last_detection_time) >= REDETECT_INTERVAL):
                force_detect = False
                dets = run_detection(model, frame)
                if dets:
                    last_detection_time = now
                    best = max(dets, key=lambda d: d["score"])
                    x1, y1, x2, y2 = map(int, best["bbox"])
                    w, h = x2 - x1, y2 - y1
                    if w > 0 and h > 0:
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, (x1, y1, w, h))
                        tracking = True

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "Detected", (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    tracking = False
                    tracker = None

            # =========================
            # TRACKING (PTZ speed mechanics)
            # =========================
            if tracking and tracker is not None:
                success, box = tracker.update(frame)
                if success:
                    mode = "tracking"
                    bbox_x, bbox_y, bbox_w, bbox_h = map(int, box)
                    tx = bbox_x + bbox_w // 2
                    ty = bbox_y + bbox_h // 2
                    offx = tx - cx0
                    offy = ty - cy0

                    target_pan_speed = clamp(PAN_KP * offx, -MAX_PAN_SPEED, MAX_PAN_SPEED)
                    target_tilt_speed = clamp(-TILT_KP * offy, -MAX_TILT_SPEED, MAX_TILT_SPEED)

                    if abs(offx) < PIX_DEADZONE:
                        target_pan_speed = 0.0
                    if abs(offy) < PIX_DEADZONE:
                        target_tilt_speed = 0.0

                    # tilt direction flip protection
                    if current_tilt_speed != 0.0 and target_tilt_speed != 0.0:
                        if (current_tilt_speed > 0) != (target_tilt_speed > 0):
                            if abs(current_tilt_speed) > 40.0:
                                target_tilt_speed = 0.0

                    current_pan_speed = accel_decel(
                        current_pan_speed, target_pan_speed,
                        MAX_PAN_ACCEL, MAX_PAN_ACCEL * 1.2, dt
                    )
                    current_tilt_speed = accel_decel(
                        current_tilt_speed, target_tilt_speed,
                        MAX_TILT_ACCEL, MAX_TILT_DECEL, dt
                    )

                    pan += current_pan_speed * dt
                    tilt += current_tilt_speed * dt

                    pan = clamp(pan, -PAN_LIMIT, PAN_LIMIT)
                    tilt = clamp(tilt, TILT_LIMIT_DOWN, TILT_LIMIT_UP)

                    # kill speed pushing into tilt limits
                    if (tilt >= TILT_LIMIT_UP and current_tilt_speed > 0):
                        current_tilt_speed = 0.0
                    if (tilt <= TILT_LIMIT_DOWN and current_tilt_speed < 0):
                        current_tilt_speed = 0.0

                    cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 255, 255), 2)
                    cv2.putText(frame, "Tracking", (bbox_x, bbox_y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    tracking = False
                    tracker = None
                    current_pan_speed = 0.0
                    current_tilt_speed = 0.0
                    ptz.stop()
                    mode = "lost"

            # =========================
            # PTZ SEND (rate-limited)
            # =========================
            if (now - last_send) >= send_dt:
                pan_changed = abs(pan - last_sent_pan) >= MIN_DEG_STEP
                tilt_changed = abs(tilt - last_sent_tilt) >= (MIN_DEG_STEP * 0.5)

                if pan_changed or tilt_changed:
                    ptz.move_abs(pan, tilt)
                    last_sent_pan = pan
                    last_sent_tilt = tilt

                last_send = now

            # =========================
            # UI overlays
            # =========================
            dist = get_laser_distance()
            if dist is not None:
                cv2.putText(frame, f"Laser: {dist:.2f} m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            h, w = frame.shape[:2]
            cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
            cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 0), 1)

            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, time.time() - t0))
            cv2.putText(frame, f"Mode: {mode}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.2f}", (w - 140, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"PTZ pan={pan:.0f} tilt={tilt:.0f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            if not tracking:
                cv2.putText(frame, "Manual: WASD move | z zero | k stop | SPACE force-detect | r reset", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

            # =========================
            # CSV log
            # =========================
            writer.writerow([
                now, mode,
                bbox_x, bbox_y, bbox_w, bbox_h,
                tx, ty,
                offx, offy,
                pan, tilt,
                current_pan_speed, current_tilt_speed,
                dist if dist is not None else "",
                fps,
            ])

            # =========================
            # Display + keys
            # =========================
            cv2.imshow("Detection + Tracking (PTZ) + Manual Aim", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("x")):
                stop_event.set()
                continue

            # Reset tracking anytime
            if key == ord("r"):
                tracking = False
                tracker = None
                current_pan_speed = 0.0
                current_tilt_speed = 0.0
                ptz.stop()
                continue

            # Manual controls ONLY when not tracking
            if not tracking:
                if key == ord("a"):
                    pan = clamp(pan - MANUAL_PAN_STEP, -PAN_LIMIT, PAN_LIMIT)
                elif key == ord("d"):
                    pan = clamp(pan + MANUAL_PAN_STEP, -PAN_LIMIT, PAN_LIMIT)
                elif key == ord("w"):
                    tilt = clamp(tilt + MANUAL_TILT_STEP, TILT_LIMIT_DOWN, TILT_LIMIT_UP)
                elif key == ord("s"):
                    tilt = clamp(tilt - MANUAL_TILT_STEP, TILT_LIMIT_DOWN, TILT_LIMIT_UP)
                elif key == ord("k"):
                    current_pan_speed = 0.0
                    current_tilt_speed = 0.0
                    ptz.stop()
                elif key == ord("z"):
                    ptz.set_zero()
                    time.sleep(0.2)
                    pan, tilt = 0.0, 0.0
                    current_pan_speed = 0.0
                    current_tilt_speed = 0.0
                    ptz.move_abs(pan, tilt)
                    last_sent_pan = pan
                    last_sent_tilt = tilt
                elif key == ord(" "):
                    # force immediate detection next loop
                    force_detect = True

    finally:
        stop_event.set()
        try:
            ptz.stop()
        except Exception:
            pass
        try:
            ptz.close()
        except Exception:
            pass

        try:
            picam2.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

        try:
            log_file.close()
        except Exception:
            pass

        print("[INFO] Finished. Log saved to tracking_log.csv")


if __name__ == "__main__":
    main()