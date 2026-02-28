#!/usr/bin/env python3
import sys
import time
import json
import re
import math
import csv
import threading
import queue
import signal

import cv2
import numpy as np

import degirum as dg
from picamera2 import Picamera2

import serial

from laser_reader_thread import start_laser_reader, get_laser_distance

# ---- matplotlib offscreen for PiP
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# =========================
# CONFIG
# =========================
ACOUSTIC_SERIAL_PORT = "/dev/ttyUSB0"
ACOUSTIC_BAUD = 115200

# NEW PTZ (USB firmware)
GIMBAL_USB_PORT = "/dev/ttyUSB0"
GIMBAL_BAUD = 115200
GIMBAL_INIT_SLEEP_S = 2.0

SET_ZERO_ON_START = True

# Acoustic "front" reference:
FRONT_DEG = 110.0
EL_DEG = 0.0

# Acoustic aiming restrictions
ACOUSTIC_AZ_LIMIT = 70.0
ACOUSTIC_EL_LIMIT = 70.0

# Ignore behind: if abs(relative_azimuth_from_front) > IGNORE_BEHIND_DEG => ignore reading
IGNORE_BEHIND_DEG = 90.0  # recommended 90. 180 disables ignore.

# Visual
FRAME_SIZE = (640, 640)  # no tiling
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
REDETECT_INTERVAL = 0.5
LOST_HOLD_TIME = 1.0

# NEW PTZ mechanical limits (from your fast PTZ script)
PAN_LIMIT = 2880.0
TILT_LIMIT_UP = 900.0
TILT_LIMIT_DOWN = -900.0

# Tracking tuning (speed-based)
PIX_DEADZONE = 5

# PID-like proportional to speed (deg/s per pixel)
PAN_KP = 2.0
TILT_KP = 2.0

MAX_PAN_SPEED = 570.0
MAX_PAN_ACCEL = 1500.0

MAX_TILT_SPEED = 220.0
MAX_TILT_ACCEL = 350.0
MAX_TILT_DECEL = 700.0

PTZ_SEND_HZ = 60.0
MIN_DEG_STEP = 0.5

# PiP acoustic 3D overlay
PIP_SIZE = (220, 220)        # (W,H)
PIP_MARGIN = 10
PIP_CORNER = "top_right"     # top_left, top_right, bottom_left, bottom_right
PIP_UPDATE_INTERVAL = 0.10   # seconds

DEBUG_ACOUSTIC = False


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
JSON_RE = re.compile(r"\{.*\}")

def extract_json_payload(line: str):
    if not line:
        return None
    m = JSON_RE.search(line)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0

def rel_az_signed(raw_az: float) -> float:
    """Signed relative azimuth around FRONT_DEG. 0=front, +left, -right."""
    return wrap180(raw_az - FRONT_DEG)

def rel_el_signed(raw_el: float) -> float:
    return wrap180(raw_el - EL_DEG)


# =========================
# NEW PTZ USB API
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
        # firmware expects absolute commands
        self.send(f"P{pan:.2f}")
        self.send(f"T{tilt:.2f}")

    def close(self):
        self.ser.close()


def accel_decel(current: float, target: float, accel_up: float, accel_down: float, dt: float) -> float:
    """
    accel_up   = max accel when speeding up in same direction (deg/s^2)
    accel_down = max decel when slowing down / reversing (deg/s^2)
    """
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
# Acoustic -> PTZ absolute targets
# =========================
def acoustic_to_ptz_targets(raw_az: float, raw_el: float):
    """
    Keep your original semantics:
      - compute relative az/el around FRONT_DEG/EL_DEG
      - ignore behind
      - clamp to acoustic limits
      - map to pan/tilt targets
    Then clamp to PTZ firmware limits.
    """
    az = rel_az_signed(raw_az)     # +left, -right
    el = rel_el_signed(raw_el)

    if abs(az) > IGNORE_BEHIND_DEG:
        return None, az, el

    az_c = clamp(az, -ACOUSTIC_AZ_LIMIT, ACOUSTIC_AZ_LIMIT)
    el_c = clamp(el, -ACOUSTIC_EL_LIMIT, ACOUSTIC_EL_LIMIT)

    # Map: az>0 means left, but PTZ pan>0 assumed right => invert
    pan_target = -az_c
    tilt_target = el_c

    pan_target = clamp(pan_target, -PAN_LIMIT, PAN_LIMIT)
    tilt_target = clamp(tilt_target, TILT_LIMIT_DOWN, TILT_LIMIT_UP)

    return (pan_target, tilt_target), az, el


# =========================
# 3D VECTOR (for PiP)
# =========================
def azimuth_to_relative_deg(raw_azimuth_deg: float) -> float:
    return (raw_azimuth_deg - FRONT_DEG) % 360.0

def az_el_to_vector(rel_az_deg: float, elev_deg: float):
    az = math.radians(rel_az_deg)
    el = math.radians(elev_deg)
    c = math.cos(el)
    z = math.sin(el)
    x = -math.sin(az) * c
    y =  math.cos(az) * c
    return x, y, z


# =========================
# Acoustic reader thread
# =========================
acoustic_q = queue.Queue(maxsize=200)

latest_lock = threading.Lock()
latest_acoustic = None  # for PiP

def acoustic_reader():
    try:
        ser = serial.Serial(ACOUSTIC_SERIAL_PORT, ACOUSTIC_BAUD, timeout=0.1)
    except Exception as e:
        print(f"Failed to open acoustic serial {ACOUSTIC_SERIAL_PORT}: {e}", file=sys.stderr)
        return

    print(f"[INFO] Acoustic serial opened: {ACOUSTIC_SERIAL_PORT} @ {ACOUSTIC_BAUD}")

    try:
        while not stop_event.is_set():
            line = ser.readline()
            if not line:
                continue
            s = line.decode("utf-8", errors="replace").strip()
            payload = extract_json_payload(s)
            if not payload:
                continue

            if DEBUG_ACOUSTIC:
                print("[ACOUSTIC]", payload)

            # queue (freshest)
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

            # latest for PiP
            with latest_lock:
                global latest_acoustic
                latest_acoustic = payload

    finally:
        try:
            ser.close()
        except Exception:
            pass


# =========================
# Detection helpers (no tiling)
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
# Matplotlib 3D PiP renderer
# =========================
class AcousticPipRenderer:
    def __init__(self, size_px=(220, 220)):
        self.w, self.h = size_px
        dpi = 100
        fig_w = self.w / dpi
        fig_h = self.h / dpi

        self.fig = plt.Figure(figsize=(fig_w, fig_h), dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.set_axis_off()
        self.ax.view_init(elev=22, azim=-55)

        u = np.linspace(0, 2*np.pi, 28)
        v = np.linspace(0, np.pi, 14)
        uu, vv = np.meshgrid(u, v)
        X = np.cos(uu) * np.sin(vv)
        Y = np.sin(uu) * np.sin(vv)
        Z = np.cos(vv)
        self.ax.plot_wireframe(X, Y, Z, linewidth=0.4, alpha=0.25)

        self.ax.plot([0, 0],   [0, 1.2], [0, 0], linewidth=1.2)   # forward
        self.ax.plot([0, 1.2], [0, 0],   [0, 0], linewidth=1.2)   # right
        self.ax.plot([0, 0],   [0, 0],   [0, 1.2], linewidth=1.2) # up
        self.ax.text(0, 1.15, 0, "F", fontsize=8)
        self.ax.text(1.15, 0, 0, "R", fontsize=8)
        self.ax.text(0, 0, 1.15, "U", fontsize=8)

        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_zlim(-1.2, 1.2)

        self.qv = self.ax.quiver(0, 0, 0, 0, 1, 0, length=1.0, normalize=True)
        self.pt = self.ax.scatter([0], [1], [0], s=18)

    def render(self, vec_xyz, label_text=""):
        x, y, z = vec_xyz

        try:
            self.qv.remove()
        except Exception:
            pass
        self.qv = self.ax.quiver(0, 0, 0, x, y, z, length=1.0, normalize=True)
        self.pt._offsets3d = ([x], [y], [z])

        self.ax.set_title(label_text, fontsize=8, pad=2)
        self.canvas.draw()

        buf = np.frombuffer(self.canvas.tostring_rgb(), dtype=np.uint8)
        img = buf.reshape(self.h, self.w, 3)  # RGB
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img_bgr


def overlay_pip(frame_bgr, pip_bgr, corner="top_right", margin=10):
    fh, fw = frame_bgr.shape[:2]
    ph, pw = pip_bgr.shape[:2]

    if corner == "top_left":
        x0, y0 = margin, margin
    elif corner == "top_right":
        x0, y0 = fw - pw - margin, margin
    elif corner == "bottom_left":
        x0, y0 = margin, fh - ph - margin
    else:  # bottom_right
        x0, y0 = fw - pw - margin, fh - ph - margin

    x0 = int(clamp(x0, 0, fw - pw))
    y0 = int(clamp(y0, 0, fh - ph))

    frame_bgr[y0:y0+ph, x0:x0+pw] = pip_bgr
    return frame_bgr


# =========================
# MAIN
# =========================
def main():
    # Start acoustic thread
    t_ac = threading.Thread(target=acoustic_reader, daemon=True)
    t_ac.start()

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

    # Camera (adjust index if needed)
    picam2 = Picamera2(0)
    picam2.configure(
        picam2.create_preview_configuration(main={"format": "RGB888", "size": FRAME_SIZE})
    )
    picam2.start()

    # PTZ
    ptz = UsbPTZ(GIMBAL_USB_PORT, GIMBAL_BAUD)
    print("[INFO] PTZ ID:", ptz.identify())
    if SET_ZERO_ON_START:
        ptz.set_zero()

    # Absolute pose + speeds (speed-based tracking like your 2nd code)
    pan = 0.0
    tilt = 0.0
    current_pan_speed = 0.0
    current_tilt_speed = 0.0

    last_send = 0.0
    send_dt = 1.0 / PTZ_SEND_HZ
    last_sent_pan = pan
    last_sent_tilt = tilt

    # Tracking state (same logic as your 1st code)
    tracker = None
    tracking = False
    last_detection_time = 0.0
    last_track_lost_time = 0.0

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
        "raw_az", "raw_el",
        "rel_az", "rel_el"
    ])

    # PiP
    pip = AcousticPipRenderer(size_px=PIP_SIZE)
    last_pip_update = 0.0
    vec_xyz = (0.0, 1.0, 0.0)
    pip_img = pip.render(vec_xyz, "waiting")

    fps = 0.0

    # For CSV
    last_raw_az = None
    last_raw_el = None
    last_rel_az = None
    last_rel_el = None

    t_prev = time.time()

    try:
        while not stop_event.is_set():
            t0 = time.time()
            now = time.time()

            # dt for speed integration (limit to avoid huge jumps)
            dt = now - t_prev
            t_prev = now
            dt = clamp(dt, 0.0, 0.05)

            # Capture
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            mode = "acoustic"
            bbox_x = bbox_y = bbox_w = bbox_h = -1
            tx = ty = -1
            offx = offy = 0

            # =========================
            # DETECT periodically (same idea)
            # =========================
            if (now - last_detection_time) >= REDETECT_INTERVAL:
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
                        last_track_lost_time = 0.0

                        # (old IO flash removed - PTZ firmware has no IO here)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "Detected", (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    tracking = False
                    tracker = None
                    last_track_lost_time = now

            # =========================
            # TRACKING (NEW mechanics)
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

                    # target speed (deg/s)
                    target_pan_speed = clamp(PAN_KP * offx, -MAX_PAN_SPEED, MAX_PAN_SPEED)
                    target_tilt_speed = clamp(-TILT_KP * offy, -MAX_TILT_SPEED, MAX_TILT_SPEED)

                    # deadzone
                    if abs(offx) < PIX_DEADZONE:
                        target_pan_speed = 0.0
                    if abs(offy) < PIX_DEADZONE:
                        target_tilt_speed = 0.0

                    # optional "force zero-crossing" for tilt direction flips (from your PTZ script)
                    if current_tilt_speed != 0.0 and target_tilt_speed != 0.0:
                        if (current_tilt_speed > 0) != (target_tilt_speed > 0):
                            if abs(current_tilt_speed) > 40.0:
                                target_tilt_speed = 0.0

                    # accel/decel limiting
                    current_pan_speed = accel_decel(
                        current_pan_speed, target_pan_speed,
                        MAX_PAN_ACCEL, MAX_PAN_ACCEL * 1.2, dt
                    )
                    current_tilt_speed = accel_decel(
                        current_tilt_speed, target_tilt_speed,
                        MAX_TILT_ACCEL, MAX_TILT_DECEL, dt
                    )

                    # integrate to absolute pose
                    pan += current_pan_speed * dt
                    tilt += current_tilt_speed * dt

                    pan = clamp(pan, -PAN_LIMIT, PAN_LIMIT)
                    tilt = clamp(tilt, TILT_LIMIT_DOWN, TILT_LIMIT_UP)

                    # if we hit limits, kill speed pushing into the limit
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
                    last_track_lost_time = time.time()
                    current_pan_speed = 0.0
                    current_tilt_speed = 0.0
                    ptz.stop()

            # =========================
            # NOT TRACKING: HOLD
            # =========================
            if not tracking:
                # if recently lost, hold for LOST_HOLD_TIME
                if last_track_lost_time and ((time.time() - last_detection_time) < LOST_HOLD_TIME):
                    mode = "hold"
                    # keep PTZ as-is (no new commands here), try re-detect (same as your code)
                    if (now - last_detection_time) >= REDETECT_INTERVAL:
                        dets = run_detection(model, frame)
                        now = time.time()
                        if dets:
                            last_detection_time = now
                            best = max(dets, key=lambda d: d["score"])
                            x1, y1, x2, y2 = map(int, best["bbox"])
                            w, h = x2 - x1, y2 - y1
                            if w > 0 and h > 0:
                                tracker = cv2.TrackerCSRT_create()
                                tracker.init(frame, (x1, y1, w, h))
                                tracking = True
                                last_track_lost_time = 0.0

                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, "Detected", (x1, y1 - 8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # =========================
                # ACOUSTIC MODE (after hold)
                # =========================
                if (not tracking) and ((time.time() - last_detection_time) >= LOST_HOLD_TIME):
                    mode = "acoustic"

                    # when switching to acoustic control, stop speeds (so tracking inertia doesn't keep integrating)
                    current_pan_speed = 0.0
                    current_tilt_speed = 0.0

                    # Drain queue, keep newest
                    newest = None
                    while True:
                        try:
                            newest = acoustic_q.get_nowait()
                        except queue.Empty:
                            break

                    if newest is not None:
                        try:
                            last_raw_az = float(newest.get("azimuth", 0.0))
                            last_raw_el = float(newest.get("elevation", 0.0))

                            # keep your "-10" elevation tweak
                            targets, ra, re = acoustic_to_ptz_targets(last_raw_az, last_raw_el - 10.0)
                            last_rel_az = ra
                            last_rel_el = re

                            if targets is not None:
                                pan, tilt = targets
                                pan = clamp(pan, -PAN_LIMIT, PAN_LIMIT)
                                tilt = clamp(tilt, TILT_LIMIT_DOWN, TILT_LIMIT_UP)
                                # send will happen via rate-limited block below
                        except Exception:
                            pass

                    # Run detection periodically (same)
                    if (now - last_detection_time) >= REDETECT_INTERVAL:
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
                                last_track_lost_time = 0.0

                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, "Detected", (x1, y1 - 8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # =========================
            # PTZ SEND (rate-limited, only if moved enough)
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
            # PiP UPDATE (draw acoustic 3D)
            # =========================
            if (now - last_pip_update) >= PIP_UPDATE_INTERVAL:
                last_pip_update = now
                with latest_lock:
                    p = latest_acoustic

                if p is not None:
                    try:
                        azp = float(p.get("azimuth", 0.0))
                        elp = float(p.get("elevation", 0.0))
                        rel_az_vis = azimuth_to_relative_deg(azp)
                        vec_xyz = az_el_to_vector(rel_az_vis, elp)
                        label = f"az {wrap180(azp-FRONT_DEG):.0f} | el {elp:.0f}"
                    except Exception:
                        vec_xyz = (0.0, 1.0, 0.0)
                        label = "bad data"
                else:
                    label = "waiting"

                pip_img = pip.render(vec_xyz, label)

            if pip_img is not None and (pip_img.shape[1], pip_img.shape[0]) != PIP_SIZE:
                pip_img = cv2.resize(pip_img, PIP_SIZE)

            if pip_img is not None:
                overlay_pip(frame, pip_img, corner=PIP_CORNER, margin=PIP_MARGIN)

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

            # =========================
            # CSV log (one line per frame)
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
                last_raw_az if last_raw_az is not None else "",
                last_raw_el if last_raw_el is not None else "",
                last_rel_az if last_rel_az is not None else "",
                last_rel_el if last_rel_el is not None else "",
            ])

            # =========================
            # Display + exit
            # =========================
            cv2.imshow("Detection System (Acoustic + Tracking + 3D PiP) [PTZ]", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("x")):
                stop_event.set()

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