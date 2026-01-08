#!/usr/bin/env python3
"""
Unified detection system + Acoustic 3D PiP overlay.

- Acoustic serial JSON (azimuth/elevation) guides gimbal ONLY when not tracking.
- Visual detection (DeGirum) initializes CSRT tracker.
- Visual tracking drives gimbal via PID (acoustic ignored while tracking).
- If tracking is lost: hold last position for 1 second; if no reacquire -> return to acoustic.
- Laser distance overlay + CSV logging.
- Adds a small picture-in-picture 3D acoustic view (unit sphere + direction arrow).

Acoustic restriction logic:
- Compute signed relative azimuth/elevation around "front".
- Clamp to ±45° for azimuth and elevation (front cone).
- If you want *ignore behind hemisphere*, set IGNORE_BEHIND_DEG=90 (recommended).
"""

import sys
import time
import json
import re
import threading
import queue
import csv
import math

import cv2
import numpy as np

import degirum as dg
from picamera2 import Picamera2

from base_ctrl import BaseController
from laser_reader_thread import start_laser_reader, get_laser_distance

# ---- matplotlib offscreen 3D render (PiP overlay)
import matplotlib
matplotlib.use("Agg")  # IMPORTANT: offscreen rendering
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# ----------------------------
# Config
# ----------------------------
ACOUSTIC_SERIAL_PORT = "/dev/ttyUSB0"
ACOUSTIC_BAUD = 115200

GIMBAL_SERIAL_PORT = "/dev/ttyAMA0"
GIMBAL_BAUD = 115200

FRONT_DEG = 120.0  # raw azimuth that corresponds to "forward"

# Allowed acoustic cone around front
ACOUSTIC_AZ_LIMIT = 45.0
ACOUSTIC_EL_LIMIT = 45.0

# Optional: ignore acoustic readings beyond this angle from front (behind hemisphere = 90)
# Set to 180 to "never ignore" (but still clamp to ±45). Recommended: 90.
IGNORE_BEHIND_DEG = 90.0

# Visual pipeline
FRAME_SIZE = (640, 640)  # no tiling
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
REDETECT_INTERVAL = 0.5
LOST_HOLD_TIME = 1.0

# Servo limits
PAN_LIMIT = 180
TILT_LIMIT_UP = 90
TILT_LIMIT_DOWN = -30

PIX_DEADZONE = 5

# PiP overlay
PIP_SIZE = (220, 220)       # width, height in pixels
PIP_MARGIN = 10             # margin from corner
PIP_CORNER = "top_right"    # "top_left" / "top_right" / "bottom_left" / "bottom_right"

DEBUG_ACOUSTIC = False


# ----------------------------
# JSON extraction
# ----------------------------
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


# ----------------------------
# Angle + vector math (matching your first script)
# ----------------------------
def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def azimuth_to_relative_deg(raw_azimuth_deg: float) -> float:
    # 0° = FRONT, CCW positive
    return (raw_azimuth_deg - FRONT_DEG) % 360.0

def az_el_to_vector(rel_az_deg: float, elev_deg: float):
    """
    Drawing coordinates:
      +Y = forward
      +X = right
      +Z = up

    rel_az_deg:
      0°   = +Y (forward)
      90°  = left (CCW)
      180° = back
      270° = right

    elev_deg:
      0° horizon, +90° up
    """
    az = math.radians(rel_az_deg)
    el = math.radians(elev_deg)

    c = math.cos(el)
    z = math.sin(el)

    # CCW positive from +Y => left becomes -X
    x = -math.sin(az) * c
    y =  math.cos(az) * c
    return x, y, z

def acoustic_to_gimbal_targets(raw_az: float, raw_el: float):
    """
    Convert raw acoustic azimuth/elevation to gimbal pan/tilt targets:
    1) signed around front in [-180..180)
    2) ignore if abs(rel) > IGNORE_BEHIND_DEG
    3) clamp to ±45 (front cone)
    """
    rel_az_signed = wrap180(raw_az - FRONT_DEG)  # +left, -right
    rel_el_signed = raw_el  # if your elevation is already centered around 0; otherwise adapt here.

    if abs(rel_az_signed) > IGNORE_BEHIND_DEG:
        return None
    if abs(rel_el_signed) > 180.0:  # safety
        return None

    rel_az_signed = clamp(rel_az_signed, -ACOUSTIC_AZ_LIMIT, ACOUSTIC_AZ_LIMIT)
    rel_el_signed = clamp(rel_el_signed, -ACOUSTIC_EL_LIMIT, ACOUSTIC_EL_LIMIT)

    # Map: rel_az_signed positive means left, but servo pan positive assumed right => invert
    pan_target = -rel_az_signed
    tilt_target = rel_el_signed

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
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        return clamp(out, -self.output_limit, self.output_limit)


# ----------------------------
# Acoustic reader thread
# ----------------------------
acoustic_q = queue.Queue(maxsize=100)
latest_acoustic_lock = threading.Lock()
latest_acoustic = None  # store the newest payload for PiP rendering

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

            if DEBUG_ACOUSTIC:
                print("[ACOUSTIC]", payload)

            # keep freshest in queue
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

            # store newest for PiP
            with latest_acoustic_lock:
                global latest_acoustic
                latest_acoustic = payload

    finally:
        try:
            ser.close()
        except Exception:
            pass


# ----------------------------
# Detection helpers
# ----------------------------
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

def run_detection(model, frame):
    res = model.predict(frame)
    dets = []
    for det in res.results:
        if det.get("score", 0.0) >= CONF_THRESHOLD:
            dets.append(det)
    return apply_nms_xyxy(dets, IOU_THRESHOLD)


# ----------------------------
# Matplotlib 3D PiP renderer
# ----------------------------
class AcousticPipRenderer:
    def __init__(self, size_px=(220, 220)):
        self.w, self.h = size_px

        # small figure sized for pixels
        dpi = 100
        fig_w = self.w / dpi
        fig_h = self.h / dpi

        self.fig = plt.Figure(figsize=(fig_w, fig_h), dpi=dpi)
        self.canvas = FigureCanvas(self.fig)

        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_axis_off()
        self.ax.view_init(elev=22, azim=-55)

        # Draw unit sphere wireframe
        u = np.linspace(0, 2*np.pi, 28)
        v = np.linspace(0, np.pi, 14)
        uu, vv = np.meshgrid(u, v)
        X = np.cos(uu) * np.sin(vv)
        Y = np.sin(uu) * np.sin(vv)
        Z = np.cos(vv)
        self.ax.plot_wireframe(X, Y, Z, linewidth=0.4, alpha=0.25)

        # Axes reference (forward/right/up)
        self.ax.plot([0, 0],   [0, 1.2], [0, 0], linewidth=1.2)   # forward
        self.ax.plot([0, 1.2], [0, 0],   [0, 0], linewidth=1.2)   # right
        self.ax.plot([0, 0],   [0, 0],   [0, 1.2], linewidth=1.2) # up

        self.ax.text(0, 1.15, 0, "F", fontsize=8)
        self.ax.text(1.15, 0, 0, "R", fontsize=8)
        self.ax.text(0, 0, 1.15, "U", fontsize=8)

        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_zlim(-1.2, 1.2)

        # Initial arrow
        self.qv = self.ax.quiver(0, 0, 0, 0, 1, 0, length=1.0, normalize=True)
        self.pt = self.ax.scatter([0], [1], [0], s=18)

        self.last_img = None

    def render(self, vec_xyz, label_text=None):
        """Return BGR uint8 image of the 3D view."""
        x, y, z = vec_xyz

        # Update arrow
        try:
            self.qv.remove()
        except Exception:
            pass
        self.qv = self.ax.quiver(0, 0, 0, x, y, z, length=1.0, normalize=True)
        self.pt._offsets3d = ([x], [y], [z])

        if label_text:
            self.ax.set_title(label_text, fontsize=8, pad=2)
        else:
            self.ax.set_title("", fontsize=8, pad=2)

        # Draw canvas
        self.canvas.draw()

        # Convert to numpy
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

    # bounds check
    x0 = clamp(x0, 0, fw - pw)
    y0 = clamp(y0, 0, fh - ph)

    # simple overlay (no alpha)
    frame_bgr[y0:y0+ph, x0:x0+pw] = pip_bgr
    return frame_bgr


# ----------------------------
# Main
# ----------------------------
def main():
    # Start acoustic thread
    threading.Thread(target=acoustic_reader, daemon=True).start()

    # Start laser reader
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
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(main={"format": "RGB888", "size": FRAME_SIZE})
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
        "mode",  # acoustic / hold / tracking
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "target_cx", "target_cy",
        "offset_x", "offset_y",
        "pan_angle", "tilt_angle",
        "laser_m",
        "fps",
        "raw_az", "raw_el",
        "rel_az", "rel_el"
    ])

    fps = 0.0

    # PiP renderer
    pip = AcousticPipRenderer(size_px=PIP_SIZE)
    last_pip_update = 0.0
    pip_update_interval = 0.10  # seconds; adjust if you need more FPS

    # latest vector defaults
    vec_xyz = (0.0, 1.0, 0.0)
    raw_az = None
    raw_el = None
    rel_az = None
    rel_el = None

    try:
        while True:
            t0 = time.time()
            frame = picam2.capture_array()  # RGB888 from PiCam
            # OpenCV expects BGR for drawing
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            now = time.time()
            mode = "acoustic"

            # ----------------------------
            # TRACKING
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

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(frame, "Tracking", (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    tracking = False
                    tracker = None
                    last_track_lost_time = now

            # ----------------------------
            # NOT TRACKING: HOLD THEN ACOUSTIC
            # ----------------------------
            if not tracking:
                # Hold last position for LOST_HOLD_TIME after losing tracker
                if last_track_lost_time and (now - last_track_lost_time) < LOST_HOLD_TIME:
                    mode = "hold"
                    if (now - last_detection_time) >= REDETECT_INTERVAL:
                        dets = run_detection(model, frame)
                        last_detection_time = now
                        if dets:
                            best = max(dets, key=lambda d: d["score"])
                            x1, y1, x2, y2 = map(int, best["bbox"])
                            ww, hh = x2 - x1, y2 - y1
                            if ww > 0 and hh > 0:
                                tracker = cv2.TrackerCSRT_create()
                                tracker.init(frame, (x1, y1, ww, hh))
                                tracking = True
                                last_track_lost_time = 0.0

                                base.send_command({"T": 132, "IO4": 0, "IO5": 255})
                                base.send_command({"T": 132, "IO4": 0, "IO5": 0})

                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, "Detected", (x1, y1 - 8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # After hold window -> acoustic + detection
                if not tracking and ((not last_track_lost_time) or (now - last_track_lost_time) >= LOST_HOLD_TIME):
                    mode = "acoustic"

                    # freshest acoustic payload
                    newest = None
                    while True:
                        try:
                            newest = acoustic_q.get_nowait()
                        except queue.Empty:
                            break

                    if newest is not None:
                        try:
                            raw_az = float(newest.get("azimuth", 0.0))
                            raw_el = float(newest.get("elevation", 0.0))

                            # Signed relative angles (for logging)
                            rel_az = wrap180(raw_az - FRONT_DEG)
                            rel_el = raw_el

                            targets = acoustic_to_gimbal_targets(raw_az, raw_el)
                            if targets is not None:
                                pan_angle, tilt_angle = targets
                                base.gimbal_ctrl(pan_angle, tilt_angle, 0, 0)
                        except Exception:
                            pass

                    # detection periodically
                    if (now - last_detection_time) >= REDETECT_INTERVAL:
                        dets = run_detection(model, frame)
                        last_detection_time = now
                        if dets:
                            best = max(dets, key=lambda d: d["score"])
                            x1, y1, x2, y2 = map(int, best["bbox"])
                            ww, hh = x2 - x1, y2 - y1
                            if ww > 0 and hh > 0:
                                tracker = cv2.TrackerCSRT_create()
                                tracker.init(frame, (x1, y1, ww, hh))
                                tracking = True
                                last_track_lost_time = 0.0

                                base.send_command({"T": 132, "IO4": 0, "IO5": 255})
                                base.send_command({"T": 132, "IO4": 0, "IO5": 0})

                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, "Detected", (x1, y1 - 8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ----------------------------
            # Update PiP acoustic 3D view (even while tracking)
            # ----------------------------
            if (now - last_pip_update) >= pip_update_interval:
                last_pip_update = now

                # take latest acoustic payload for visualization
                with latest_acoustic_lock:
                    p = latest_acoustic

                if p is not None:
                    try:
                        raw_az_vis = float(p.get("azimuth", 0.0))
                        el_vis = float(p.get("elevation", 0.0))
                        rel_az_vis = azimuth_to_relative_deg(raw_az_vis)
                        vec_xyz = az_el_to_vector(rel_az_vis, el_vis)
                        label = f"az {wrap180(raw_az_vis-FRONT_DEG):.0f} | el {el_vis:.0f}"
                    except Exception:
                        vec_xyz = (0.0, 1.0, 0.0)
                        label = "no data"
                else:
                    label = "waiting"

                pip_img = pip.render(vec_xyz, label_text=label)
            else:
                pip_img = pip.last_img if pip.last_img is not None else pip.render(vec_xyz, label_text="")

            # Ensure PiP size (in case)
            if pip_img.shape[1] != PIP_SIZE[0] or pip_img.shape[0] != PIP_SIZE[1]:
                pip_img = cv2.resize(pip_img, PIP_SIZE)

            # Overlay PiP
            overlay_pip(frame, pip_img, corner=PIP_CORNER, margin=PIP_MARGIN)

            # ----------------------------
            # Overlays: laser, crosshair, mode, fps
            # ----------------------------
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

            # ----------------------------
            # Logging
            # ----------------------------
            if tracking and tracker is not None:
                # last tracker box drawn above; log bbox if available
                # (when tracking just started, box is drawn by detection too)
                pass

            # log one line per frame
            # For tracking mode, bbox/offset are only meaningful when tracker succeeded in this frame.
            # We'll store -1 when unknown.
            writer.writerow([
                now,
                mode,
                -1, -1, -1, -1,
                -1, -1,
                0, 0,
                pan_angle, tilt_angle,
                dist,
                fps,
                raw_az if raw_az is not None else "",
                raw_el if raw_el is not None else "",
                rel_az if rel_az is not None else "",
                rel_el if rel_el is not None else "",
            ])

            # Show
            cv2.imshow("Detection System (Acoustic + PiP 3D)", frame)
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
