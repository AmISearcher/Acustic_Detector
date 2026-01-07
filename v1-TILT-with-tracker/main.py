import degirum as dg
import cv2
import numpy as np
import time
from picamera2 import Picamera2
from base_ctrl import BaseController
import csv
from laser_reader_thread import start_laser_reader, get_laser_distance

# Load model
model = dg.load_model(
    model_name="best",
    inference_host_address="@local",
    zoo_url="../models/models_640_resized_hq",
    token="",
    overlay_color=(0, 255, 0)
)

class PID:
    def __init__(self, kp, ki, kd, output_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.prev_error = 0
        self.integral = 0

    def compute(self, error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(min(output, self.output_limit), -self.output_limit)

def boxes_overlap(boxA, boxB, threshold=0.2):
    ax1, ay1, aw, ah = boxA
    bx1, by1, bw, bh = boxB
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    return (intersection / union) >= threshold if union else False

def generate_tiles(frame):
    h, w, _ = frame.shape
    tiles, coords = [], []
    for y in range(0, h - TILE_HEIGHT + 1, STRIDE_Y):
        for x in range(0, w - TILE_WIDTH + 1, STRIDE_X):
            tiles.append(frame[y:y + TILE_HEIGHT, x:x + TILE_WIDTH])
            coords.append((x, y))
    return tiles, coords

def apply_nms(detections, iou_thresh=0.5):
    if not detections:
        return []
    boxes = np.array([d["bbox"] for d in detections])
    scores = np.array([d["score"] for d in detections])
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF_THRESHOLD, iou_thresh)
    if isinstance(indices, np.ndarray):
        indices = indices.flatten().tolist()
    elif isinstance(indices, list) and isinstance(indices[0], (list, tuple)):
        indices = [i[0] for i in indices]
    return [detections[i] for i in indices]

def run_detection(frame):
    tiles, coords = generate_tiles(frame)
    results = model.predict_batch(tiles)
    detections = []
    for result, (dx, dy) in zip(results, coords):
        for det in result.results:
            if det["score"] >= CONF_THRESHOLD:
                det["bbox"] = [
                    det["bbox"][0] + dx,
                    det["bbox"][1] + dy,
                    det["bbox"][2] + dx,
                    det["bbox"][3] + dy,
                ]
                detections.append(det)
    return apply_nms(detections, IOU_THRESHOLD)

TILE_WIDTH, TILE_HEIGHT = 480, 480
OVERLAP_X, OVERLAP_Y = 0.2, 0.2
STRIDE_X = int(TILE_WIDTH * (1 - OVERLAP_X))
STRIDE_Y = int(TILE_HEIGHT * (1 - OVERLAP_Y))
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
REDETECT_INTERVAL = 0.5
NODETECT_INTERVAL = 1.5

RESIZED_WIDTH = 640
RESIZED_HEIGHT = 480


def main():
    base = BaseController('/dev/ttyAMA0', 115200)
    pan_angle, tilt_angle = 0.0, 0.0
    base.gimbal_ctrl(pan_angle, tilt_angle, 0, 0)

    pan_pid = PID(kp=0.005, ki=0.0, kd=0.0, output_limit=0.6)
    tilt_pid = PID(kp=0.005, ki=0.0, kd=0.0, output_limit=0.6)

    frame_center_x_s = RESIZED_WIDTH / 2
    frame_center_y_s = RESIZED_HEIGHT / 2
    pan_limit = 180
    tilt_limit_up = 90
    tilt_limit_down = -30

    tracker = None
    bbox = None
    tracking = False
    last_detection_time = 0
    last_nodetection_time = time.time()
    led_on = False
    fps = 0.0

    log_file = open("tracking_log.csv", "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["timestamp", "cx", "cy", "offset_x", "offset_y", "detected", "pan_angle", "tilt_angle", "fps"])

    start_laser_reader()

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration({"format": "RGB888", "size": (2592, 1944)}))
    picam2.start()

    try:
        while True:
            timestamp = time.time()
            frame_full = picam2.capture_array()
            frame_small = cv2.resize(frame_full, (RESIZED_WIDTH, RESIZED_HEIGHT))

            current_time = time.time()
            need_redetect = (current_time - last_detection_time >= REDETECT_INTERVAL)
            nodetect_time = (current_time - last_nodetection_time >= NODETECT_INTERVAL)

            if need_redetect:
                detections = run_detection(frame_full)
                if detections:
                    base.send_command({"T": 132, "IO4": 0, "IO5": 255})
                    base.send_command({"T": 132, "IO4": 0, "IO5": 0})
                    best_det = max(detections, key=lambda d: d["score"])
                    x1, y1, x2, y2 = map(int, best_det["bbox"])

                    scale_x = RESIZED_WIDTH / frame_full.shape[1]
                    scale_y = RESIZED_HEIGHT / frame_full.shape[0]
                    x1_s = int(x1 * scale_x)
                    y1_s = int(y1 * scale_y)
                    x2_s = int(x2 * scale_x)
                    y2_s = int(y2 * scale_y)
                    w_s, h_s = x2_s - x1_s, y2_s - y1_s

                    # Expand box size by 2x around center for better tracking
                    cx_s = x1_s + w_s // 2
                    cy_s = y1_s + h_s // 2
                    w_s *= 2
                    h_s *= 2
                    x1_s = max(0, cx_s - w_s // 2)
                    y1_s = max(0, cy_s - h_s // 2)
                    w_s = min(w_s, RESIZED_WIDTH - x1_s)
                    h_s = min(h_s, RESIZED_HEIGHT - y1_s)

                    if bbox is not None:
                        overlap = boxes_overlap(bbox, (x1_s, y1_s, w_s, h_s))
                    else:
                        overlap = False

                    if not tracking or not overlap:
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame_small, (x1_s, y1_s, w_s, h_s))
                        tracking = True
                        print("[INFO] Tracker reinitialized.")

                    bbox = (x1_s, y1_s, w_s, h_s)
                    last_detection_time = time.time()
                    last_nodetection_time = last_detection_time

            if tracking and not nodetect_time:
                success, box = tracker.update(frame_small)
                if success:
                    bbox = box
                    x, y, w, h = map(int, box)
                    cx, cy = x + w // 2, y + h // 2
                    offset_x_s = cx - frame_center_x_s
                    offset_y_s = cy - frame_center_y_s

                    scale_fx = frame_full.shape[1] / RESIZED_WIDTH
                    scale_fy = frame_full.shape[0] / RESIZED_HEIGHT
                    offset_x = offset_x_s * scale_fx
                    offset_y = offset_y_s * scale_fy

                    if abs(offset_x_s) > 5:
                        pan_angle += pan_pid.compute(offset_x)
                    if abs(offset_y_s) > 5:
                        tilt_angle += tilt_pid.compute(-offset_y)

                    pan_angle = max(min(pan_angle, pan_limit), -pan_limit)
                    tilt_angle = max(min(tilt_angle, tilt_limit_up), tilt_limit_down)

                    base.gimbal_ctrl(pan_angle, tilt_angle, 0, 0)


                    csv_writer.writerow([timestamp, cx, cy, offset_x, offset_y, 1, pan_angle, tilt_angle, fps])

                    cv2.rectangle(frame_small, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(frame_small, "Tracking", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    tracking = False
                    tracker = None
            else:
                tracking = False
                tracker = None
                pan_angle *= 0.95
                tilt_angle *= 0.95
                base.gimbal_ctrl(pan_angle, tilt_angle, 0, 0)
                csv_writer.writerow([timestamp, -1, -1, 0, 0, 0, pan_angle, tilt_angle, fps])

            if get_laser_distance() is not None:
                cv2.putText(frame_small, f"Laser: {get_laser_distance():.2f} m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            height, width = frame_small.shape[:2]
            cv2.line(frame_small, (width // 2, 0), (width // 2, height), (0, 255, 0), 1)
            cv2.line(frame_small, (0, height // 2), (width, height // 2), (0, 255, 0), 1)

            fps = 0.9 * fps + 0.1 * (1.0 / (time.time() - timestamp))
            cv2.putText(frame_small, f"FPS: {fps:.2f}", (width - 120, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Tracking UI", frame_small)
            if cv2.waitKey(1) & 0xFF in [ord("q"), ord("x")]:
                break

    finally:
        base.send_command({"T": 132, "IO4": 0, "IO5": 0})
        picam2.stop()
        cv2.destroyAllWindows()
        log_file.close()
        print("[INFO] Tracking finished. Log saved to tracking_log.csv")

if __name__ == "__main__":
    main()
