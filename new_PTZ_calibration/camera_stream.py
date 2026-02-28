#!/usr/bin/env python3

import cv2
from picamera2 import Picamera2


FRAME_SIZE = (640, 640)  # change if needed


def main():
    # Initialize camera (index 0 — change if needed)
    picam2 = Picamera2(0)

    # Configure preview stream
    picam2.configure(
        picam2.create_preview_configuration(
            main={"format": "RGB888", "size": FRAME_SIZE}
        )
    )

    picam2.start()

    try:
        while True:
            # Capture frame
            frame_rgb = picam2.capture_array()

            # Convert RGB -> BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Show frame
            cv2.imshow("Camera Stream", frame_bgr)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()