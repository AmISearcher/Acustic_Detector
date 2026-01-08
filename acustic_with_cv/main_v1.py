#!/usr/bin/env python3
"""
Unified detection system:
- Acoustic serial JSON (azimuth/elevation) guides gimbal ONLY when not tracking.
- Visual detection (DeGirum) initializes CSRT tracker.
- Visual tracking drives gimbal via PID (acoustic ignored while tracking).
- If tracking is lost: hold last position for 1 second; if no reacquire -> return to acoustic.
- Laser distance overlay + CSV logging.

Acoustic restriction logic (as you requested):
- Compute signed relative azimuth/elevation around "front".
- If |rel| > 180  => IGNORE reading (treat as behind / invalid)
- Else clamp rel to ±45 (front cone). Values between 0..180 are clamped to border 45.
"""

import json
import math
import re
import sys
import time
import threading
import queue
import csv

import cv2
import numpy as np

import degirum as dg
from picamera2 import Picamera2

from base_ctrl import BaseController
from laser_reader_thread import start_laser_reader, get_laser_distance


# ----------------------------
# Config
# ----------------------------
ACOUSTIC_SERIAL_PORT = "/dev/ttyUSB0"
ACOUSTIC_BAUD = 115200

GIMBAL_SERIAL_PORT = "/dev/ttyAMA0"
GIMBAL_BAUD = 115200

# Acoustic "front" reference: raw azimuth that corresponds to forward direction
FRONT_DEG = 120.0

# Allowed acoustic cone around front
ACOUSTIC_AZ_LIMIT = 45.0     # ±45° from front
ACOUSTIC_EL_LIMIT = 45.0     # ±45° from neutral

# Visual pipeline
FRAME_SIZE = (640, 640)      # no tiling: camera runs at model input size
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5

REDETECT_INTERVAL = 0.5      # seconds (when not tracking / searching)
LOST_HOLD_TIME = 1.0         # seconds to hold last gimbal position after losing track

# Servo mechanical limits (final clamp)
PAN_LIMIT = 180
TILT_LIMIT_UP = 90
TILT_LIMIT_DOWN = -30

# Pixel deadzone for PID (to avoid jitter)
PIX_DEADZONE = 5


# ----------------------------
# Helpers: angle wrapping + acoustic restriction
# ----------------------------
JSON_RE = re.compile(r"\{.*\}")

def extract_json_payload(line: str):
    """Extract first JSON object {...} from a line."""
    if not line:
        return None
    m = JSON_RE.search(line)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

def wrap180(deg: float) -> float:
    """Map any angle to [-180, +180)."""
    return (deg + 180.0) % 360.0 - 180.0

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def acoustic_to_gimbal_targets(raw_az: float, raw_el: float, front_deg: float):
    """
    Convert raw acoustic azimuth/elevation to gimbal pan/tilt targets with your rule:
    - Compute signed relative (front=0) in [-180..180)
    - If abs(rel) > 180 => ignore (won't happen with wrap180, but we keep your rule explicit)
    - Else clamp to ±45
    Return (pan_target, tilt_target) or None if ignored.
    """
    rel_az = wrap180(raw_az - front_deg)   # [-180..180)
    rel_el = wrap180(raw_el)               # assume elevation is also circular; if not, still safe

    # Your explicit ignore rule
    if abs(rel_az) > 180.0 or abs(rel_el) > 180.0:
        return None

    # Clamp to front cone borders
    rel_az = clamp(rel_az, -ACOUSTIC_AZ_LIMIT, ACOUSTIC_AZ_LIMIT)
    rel_el = clamp(rel_el, -ACOUSTIC_EL_LIMIT, ACOUSTIC_EL_LIMIT)

    # Mapping convention:
    # rel_az > 0 means "left" (CCW). If gimbal pan positive means "right",
    # then pan_target should be negative of rel_az.
    pan_target = -rel_az
    tilt_target = rel_el

    # Mechanical clamp too
    pan_target = clamp(pan_target, -PAN_LIMIT, PAN_LIMIT)
    tilt_target = clamp(tilt_target, TILT_LIMIT_DOWN, TILT_LIMIT_UP)

    return pan_target, tilt_target


# ----------------------------
# PID
# ----------------------------
class PID:
    def __init__(self, kp, ki, kd, output_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return clamp(output, -self.output_limit, self.output_limit)


# ----------------------------
# Acoustic reader thread -> latest payload
# ----------------------------
acoustic_q = queue.Queue(maxsize=50)

def acoustic_reader():
    try:
        import serial
    except ImportError:
        print("pyserial not installed. Install: pip install pyserial", file=sys.stderr)
        return

    try:
        ser = serial.Serial(ACOUSTIC_SERIAL_PORT, ACOUSTIC_BAUD, timeout=1)
    except Exception as e:
        print(f"Failed to open acoustic serial {ACOUSTIC_SERIAL_PORT}: {e}", file=sys.stderr)
        return

    print(f"[INFO] Acoustic serial opened: {ACOUSTIC_SERIAL_PORT} @ {ACOUSTIC_BAUD}")

    try:
        while True:
            line = ser.readline()
            if not line:
                continue
            s = line.decode("utf-8", errors="replace").strip()
            payload = extract_json_payload(s)
            if not payload:
                continue

            # keep freshest (drop old if full)
            try:
                acoustic_q.put_nowait(payload)
            except queue.Full:
                try:
                    _ = acoustic_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    acoustic_q.put_nowait(payload)
                except queue.Full:
                    pass
    finally:
        try:
            ser.close()
        except Exception:
            pass


# ----------------------------
# Detection helpers (no tiling)
# ----------------------------
def apply_nms_xyxy(detections, iou_thresh=0.5):
    """detections: list of dict with det['bbox']=[x1,y1,x2,y2], det['score']"""
    if not detections:
        return []
    boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
    scores = np.array([d["score"] for d in detections], dtype=np.float32)

    # OpenCV NMSBoxes expects [x,y,w,h]
    xywh = boxes.copy()
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    xywh[:, 0] = boxes[:, 0]
    xywh[:, 1] = boxes[:, 1]

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

def run_detection(model, frame):
    res = model.predict(frame)
    dets = []
    for det in res.results:
        if det.get("score", 0.0) >= CONF_THRESHOLD:
            dets.append(det)
    return apply_nms_xyxy(dets, IOU_THRESHOLD)


# ----------------------------
# Main
# ----------------------------
def main():
    # Start acoustic thread
    threading.Thread(target=acoustic_reader, daemon=True).start()

    # Start laser reader
    start_laser_reader()

    # Load DeGirum model
    model = dg.load_model(
        model_name="best",
        inference_host_address="@local",
        zoo_url="../models/models_640_resized_hq",
        token="",
        overlay_color=(0, 255, 0),
    )

    # Camera
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"format": "RGB888", "size": FRAME_SIZE}
        )
    )
    picam2.start()

    # Gimbal controller
    base = BaseController(GIMBAL_SERIAL_PORT, GIMBAL_BAUD)
    pan_angle, tilt_angle = 0.0, 0.0
    base.gimbal_ctrl(pan_angle, tilt_angle, 0, 0)

    # PID for tracking
    pan_pid = PID(kp=0.005, ki=0.0, kd=0.0, output_limit=0.6)
    tilt_pid = PID(kp=0.005, ki=0.0, kd=0.0, output_limit=0.6)

    # Tracking state
    tracker = None
    tracking = False
    last_detection_time = 0.0
    last_track_lost_time = 0.0

    # UI center
    cx0 = FRAME_SIZE[0] / 2
    cy0 = FRAME_SIZE[1] / 2

    # CSV log
    log_file = open("tracking_log.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "timestamp",
        "mode",                 # "acoustic" or "tracking" or "hold"
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "target_cx", "target_cy",
        "offset_x", "offset_y",
        "pan_angle", "tilt_angle",
        "laser_m",
        "fps"
    ])

    fps = 0.0

    try:
        while True:
            t0 = time.time()
            frame = picam2.capture_array()

            now = time.time()
            mode = "acoustic"

            # ----------------------------
            # TRACKING MODE
            # ----------------------------
            if tracking and tracker is not None:
                success, box = tracker.update(frame)
                if success:
                    mode = "tracking"
                    x, y, w, h = map(int, box)
                    tx = x + w // 2
                    ty = y + h // 2
                    offx = tx - cx0
                    offy = ty - cy0

                    if abs(offx) > PIX_DEADZONE:
                        pan_angle += pan_pid.compute(offx)
                    if abs(offy) > PIX_DEADZONE:
                        tilt_angle += tilt_pid.compute(-offy)

                    pan_angle = clamp(pan_angle, -PAN_LIMIT, PAN_LIMIT)
                    tilt_angle = clamp(tilt_angle, TILT_LIMIT_DOWN, TILT_LIMIT_UP)

                    base.gimbal_ctrl(pan_angle, tilt_angle, 0, 0)

                    # UI
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(frame, "Tracking", (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    # log
                    dist = get_laser_distance()
                    fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, time.time() - t0))
                    writer.writerow([now, mode, x, y, w, h, tx, ty, offx, offy,
                                     pan_angle, tilt_angle, dist, fps])
                else:
                    # lost target
                    tracking = False
                    tracker = None
                    last_track_lost_time = now

            # ----------------------------
            # NOT TRACKING: HOLD THEN ACOUSTIC
            # ----------------------------
            if not tracking:
                # Hold last position for 1 second after loss
                if last_track_lost_time and (now - last_track_lost_time) < LOST_HOLD_TIME:
                    mode = "hold"
                    # Try re-detection while holding, but throttled
                    if (now - last_detection_time) >= REDETECT_INTERVAL:
                        dets = run_detection(model, frame)
                        last_detection_time = now
                        if dets:
                            best = max(dets, key=lambda d: d["score"])
                            x1, y1, x2, y2 = map(int, best["bbox"])
                            w, h = x2 - x1, y2 - y1
                            if w > 0 and h > 0:
                                tracker = cv2.TrackerCSRT_create()
                                tracker.init(frame, (x1, y1, w, h))
                                tracking = True
                                last_track_lost_time = 0.0

                                # Blink indicator as you had
                                base.send_command({"T": 132, "IO4": 0, "IO5": 255})
                                base.send_command({"T": 132, "IO4": 0, "IO5": 0})

                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, "Detected", (x1, y1 - 8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # If still not tracking after hold window -> acoustic aiming
                if not tracking and ((not last_track_lost_time) or (now - last_track_lost_time) >= LOST_HOLD_TIME):
                    mode = "acoustic"

                    # Get freshest acoustic payload
                    newest = None
                    while True:
                        try:
                            newest = acoustic_q.get_nowait()
                        except queue.Empty:
                            break

                    # Apply acoustic only if we got a payload and it passes your restriction
                    if newest is not None:
                        try:
                            raw_az = float(newest.get("azimuth", 0.0))
                            raw_el = float(newest.get("elevation", 0.0))
                            targets = acoustic_to_gimbal_targets(raw_az, raw_el, FRONT_DEG)
                            if targets is not None:
                                pan_angle, tilt_angle = targets
                                base.gimbal_ctrl(pan_angle, tilt_angle, 0, 0)
                            # else: ignored (do nothing, keep last angles)
                        except Exception:
                            pass

                    # Run detection periodically while acoustic aiming
                    if (now - last_detection_time) >= REDETECT_INTERVAL:
                        dets = run_detection(model, frame)
                        last_detection_time = now
                        if dets:
                            best = max(dets, key=lambda d: d["score"])
                            x1, y1, x2, y2 = map(int, best["bbox"])
                            w, h = x2 - x1, y2 - y1
                            if w > 0 and h > 0:
                                tracker = cv2.TrackerCSRT_create()
                                tracker.init(frame, (x1, y1, w, h))
                                tracking = True
                                last_track_lost_time = 0.0

                                base.send_command({"T": 132, "IO4": 0, "IO5": 255})
                                base.send_command({"T": 132, "IO4": 0, "IO5": 0})

                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, "Detected", (x1, y1 - 8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Log idle/hold/acoustic with no bbox if not tracking
                if not tracking:
                    dist = get_laser_distance()
                    fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, time.time() - t0))
                    writer.writerow([now, mode, -1, -1, -1, -1, -1, -1, 0, 0,
                                     pan_angle, tilt_angle, dist, fps])

            # ----------------------------
            # UI overlays
            # ----------------------------
            dist = get_laser_distance()
            if dist is not None:
                cv2.putText(frame, f"Laser: {dist:.2f} m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # crosshair
            h, w = frame.shape[:2]
            cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
            cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 0), 1)

            # status
            cv2.putText(frame, f"Mode: {mode}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # fps
            cv2.putText(frame, f"FPS: {fps:.2f}", (w - 140, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Detection System (Acoustic -> Detect -> Track)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("x")):
                break

    finally:
        try:
            base.send_command({"T": 132, "IO4": 0, "IO5": 0})
        except Exception:
            pass
        try:
            picam2.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        log_file.close()
        print("[INFO] Finished. Log saved to tracking_log.csv")


if __name__ == "__main__":
    main()
