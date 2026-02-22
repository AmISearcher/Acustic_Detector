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

from base_ctrl import BaseController
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

GIMBAL_SERIAL_PORT = "/dev/ttyAMA0"
GIMBAL_BAUD = 115200

# Acoustic "front" reference:
FRONT_DEG = 115.0
EL_DEG = 10

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

# Servo mechanical limits
PAN_LIMIT = 180
TILT_LIMIT_UP = 90
TILT_LIMIT_DOWN = -30

# PID / deadzone
PIX_DEADZONE = 5
PAN_KP = 0.005
TILT_KP = 0.005
PID_OUTPUT_LIMIT = 0.6

# PiP acoustic 3D overlay
PIP_SIZE = (220, 220)        # (W,H)
PIP_MARGIN = 10
PIP_CORNER = "top_right"     # top_left, top_right, bottom_left, bottom_right
PIP_UPDATE_INTERVAL = 0.10   # seconds (lower = smoother, higher = faster main FPS)

# Shutdown behavior
HOLD_FLUSH_REPEATS = 10
HOLD_FLUSH_DT = 0.04
PARK_ON_EXIT = False         # True => park to (0,0) at exit

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

def acoustic_to_gimbal_targets(raw_az: float, raw_el: float):
    """
    Apply: ignore behind + clamp to ±45°, then map to gimbal pan/tilt.
    """
    az = rel_az_signed(raw_az)  # +left, -right
    #el = raw_el
    el = rel_el_signed(raw_el)                 # assuming elevation around 0; if not, adapt here.
    
    if abs(az) > IGNORE_BEHIND_DEG:
        return None, az, el

    az_c = clamp(az, -ACOUSTIC_AZ_LIMIT, ACOUSTIC_AZ_LIMIT)
    el_c = clamp(el, -ACOUSTIC_EL_LIMIT, ACOUSTIC_EL_LIMIT)

    # Map: az>0 means left, but pan>0 assumed right => invert
    pan_target = -az_c
    tilt_target = el_c

    pan_target = clamp(pan_target, -PAN_LIMIT, PAN_LIMIT)
    tilt_target = clamp(tilt_target, TILT_LIMIT_DOWN, TILT_LIMIT_UP)
    return (pan_target, tilt_target), az, el


# =========================
# 3D VECTOR (for PiP) — matches your first script
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

    def compute(self, error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        return clamp(out, -self.output_limit, self.output_limit)


# =========================
# Acoustic reader thread
# =========================
acoustic_q = queue.Queue(maxsize=200)

latest_lock = threading.Lock()
latest_acoustic = None  # for PiP

def acoustic_reader():
    try:
        import serial
    except ImportError:
        print("pyserial not installed. Install: pip install pyserial", file=sys.stderr)
        return

    try:
        # timeout so the thread can exit quickly when stop_event is set
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

        # unit sphere
        u = np.linspace(0, 2*np.pi, 28)
        v = np.linspace(0, np.pi, 14)
        uu, vv = np.meshgrid(u, v)
        X = np.cos(uu) * np.sin(vv)
        Y = np.sin(uu) * np.sin(vv)
        Z = np.cos(vv)
        self.ax.plot_wireframe(X, Y, Z, linewidth=0.4, alpha=0.25)

        # axes
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
        self.last_img = None

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
        self.last_img = img_bgr
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

    # Tracking state
    tracker = None
    tracking = False
    last_detection_time = 0.0
    last_track_lost_time = 0.0

    # Frame center
    cx0 = FRAME_SIZE[0] / 2
    cy0 = FRAME_SIZE[1] / 2

    # Logging
    log_file = open("tracking_log.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "timestamp",
        "mode",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "target_cx", "target_cy",
        "offset_x", "offset_y",
        "pan_angle", "tilt_angle",
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

    try:
        while not stop_event.is_set():
            t0 = time.time()

            # Capture RGB -> convert to BGR for OpenCV
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            now = time.time()
            mode = "acoustic"

            bbox_x = bbox_y = bbox_w = bbox_h = -1
            tx = ty = -1
            offx = offy = 0
            
            
            # Run detection periodically
            if (now - last_detection_time) >= REDETECT_INTERVAL:
                print("Need Redetect")
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

                        base.send_command({"T": 132, "IO4": 0, "IO5": 255})
                        base.send_command({"T": 132, "IO4": 0, "IO5": 0})

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "Detected", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    tracking = False
                    tracker = None
                    last_track_lost_time = now
                
            # =========================
            # TRACKING
            # =========================
            #now = time.time()
            #need_redetect = (now - last_detection_time) >= REDETECT_INTERVAL
            #if tracking and tracker is not None and need_redetect:
            else:
                print("Skip detection")
            if tracking and tracker is not None:
                success, box = tracker.update(frame)
                if success:
                    mode = "tracking"
                    bbox_x, bbox_y, bbox_w, bbox_h = map(int, box)
                    tx = bbox_x + bbox_w // 2
                    ty = bbox_y + bbox_h // 2
                    offx = tx - cx0
                    offy = ty - cy0

                    if abs(offx) > PIX_DEADZONE:
                        pan_angle += pan_pid.compute(offx)
                    if abs(offy) > PIX_DEADZONE:
                        ######## Changes ########
                        tilt_angle += tilt_pid.compute(-offy)

                    pan_angle = clamp(pan_angle, -PAN_LIMIT, PAN_LIMIT)
                    tilt_angle = clamp(tilt_angle, TILT_LIMIT_DOWN, TILT_LIMIT_UP)

                    base.gimbal_ctrl(pan_angle, tilt_angle, 300, 50)

                    cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 255, 255), 2)
                    cv2.putText(frame, "Tracking", (bbox_x, bbox_y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    tracking = False
                    tracker = None
                    last_track_lost_time = time.time()
            #else:
            #    tracking = False
            #    tracker = None
            #    last_track_lost_time = now

            # =========================
            # NOT TRACKING: HOLD
            # =========================
            if not tracking:
                if last_track_lost_time and ((time.time()- last_detection_time) < LOST_HOLD_TIME):
                    print("Hold")
                    mode = "hold"
                    # keep gimbal as-is (no new commands)
                    # try re-detect
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

                                base.send_command({"T": 132, "IO4": 0, "IO5": 255})
                                base.send_command({"T": 132, "IO4": 0, "IO5": 0})

                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, "Detected", (x1, y1 - 8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # =========================
                # ACOUSTIC MODE (after hold)
                # =========================

                if (not tracking) and ((time.time() - last_detection_time) >= LOST_HOLD_TIME):
                    print("Acoustic")
                    mode = "acoustic"

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
                            targets, ra, re = acoustic_to_gimbal_targets(last_raw_az, last_raw_el-10)
                            last_rel_az = ra
                            last_rel_el = re

                            if targets is not None:
                                pan_angle, tilt_angle = targets
                                base.gimbal_ctrl(pan_angle, tilt_angle, 300, 50)
                            # else: ignored => do nothing (hold last pose)
                        except Exception:
                            pass

                    # Run detection periodically
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

                                base.send_command({"T": 132, "IO4": 0, "IO5": 255})
                                base.send_command({"T": 132, "IO4": 0, "IO5": 0})

                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, "Detected", (x1, y1 - 8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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

            # Ensure size
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
                pan_angle, tilt_angle,
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
            cv2.imshow("Detection System (Acoustic + Tracking + 3D PiP)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("x")):
                stop_event.set()

    finally:
        # --- HARD STOP to avoid post-exit twitch ---
        stop_event.set()

        # stop “effects”
        try:
            base.send_command({"T": 132, "IO4": 0, "IO5": 0})
        except Exception:
            pass

        # override any queued motion by re-sending the same pose a few times
        try:
            if PARK_ON_EXIT:
                pan_angle, tilt_angle = 0.0, 0.0

            for _ in range(HOLD_FLUSH_REPEATS):
                base.gimbal_ctrl(pan_angle, tilt_angle, 300, 50)
                time.sleep(HOLD_FLUSH_DT)
        except Exception:
            pass

        # close camera + UI
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
