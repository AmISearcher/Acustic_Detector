import degirum as dg
import cv2
import numpy as np
import time
from picamera2 import Picamera2

# -------------------- Parameters --------------------
FRAME_SIZE = (640, 640)
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
REDETECT_INTERVAL = 3.0  # seconds

# -------------------- Load model --------------------
model = dg.load_model(
    model_name="best",
    inference_host_address="@local",
    zoo_url="../models/models_640_resized_hq",
    token="",
    overlay_color=(0, 255, 0),
)

# -------------------- Utils --------------------
def apply_nms(detections, iou_thresh=0.5):
    if not detections:
        return []

    boxes = np.array([d["bbox"] for d in detections])
    scores = np.array([d["score"] for d in detections])

    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        CONF_THRESHOLD,
        iou_thresh,
    )

    if len(indices) == 0:
        return []

    if isinstance(indices, np.ndarray):
        indices = indices.flatten().tolist()

    return [detections[i] for i in indices]


def run_detection(frame):
    start = time.time()

    result = model.predict(frame)

    detections = []
    for det in result.results:
        if det["score"] >= CONF_THRESHOLD:
            detections.append(det)

    detections = apply_nms(detections, IOU_THRESHOLD)

    print(f"Detection time: {time.time() - start:.3f}s")
    return detections


# -------------------- Main --------------------
def main():
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"format": "RGB888", "size": FRAME_SIZE}
        )
    )
    picam2.start()

    tracker = None
    tracking = False
    last_detection_time = 0.0
    fps = 0.0

    try:
        while True:
            start_time = time.time()
            frame = picam2.capture_array()

            now = time.time()
            need_redetect = (now - last_detection_time) >= REDETECT_INTERVAL

            if not tracking or need_redetect:
                detections = run_detection(frame)
                last_detection_time = now

                if detections:
                    best = max(detections, key=lambda d: d["score"])
                    x1, y1, x2, y2 = map(int, best["bbox"])
                    w, h = x2 - x1, y2 - y1

                    if w > 0 and h > 0:
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, (x1, y1, w, h))
                        tracking = True

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            "Detected",
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )
                    else:
                        tracking = False
                else:
                    tracking = False

            else:
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(
                        frame, (x, y), (x + w, y + h), (255, 0, 0), 2
                    )
                    cv2.putText(
                        frame,
                        "Tracking",
                        (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )
                else:
                    tracking = False

            # FPS
            fps = 0.9 * fps + 0.1 * (1.0 / (time.time() - start_time))
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Detection + CSRT (640x640)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("x")):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
