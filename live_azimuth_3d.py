#!/usr/bin/env python3
import json
import math
import sys
import threading
import queue
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    import serial
except ImportError:
    serial = None

# ----------------------------
# Config
# ----------------------------
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
FRONT_DEG = 120.0  # raw azimuth corresponding to "front"

# Set True to see incoming lines/payloads in terminal
DEBUG = False

JSON_RE = re.compile(r"\{.*\}")

# ----------------------------
# Angle mapping + 3D vector
# ----------------------------
def wrap360(deg: float) -> float:
    return deg % 360.0

def azimuth_to_relative_deg(raw_azimuth_deg: float) -> float:
    # 0° = FRONT, CCW positive
    return wrap360(raw_azimuth_deg - FRONT_DEG)

def az_el_to_vector(rel_az_deg: float, elev_deg: float):
    """
    Drawing coordinates:
      +Y = forward (front)
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

# ----------------------------
# Robust JSON extraction
# ----------------------------
def extract_json_payload(line: str):
    """
    Accepts:
      RECEIVED: {...}
      any garbage ... {...} ... garbage
    Extracts the first {...} block.
    """
    if not line:
        return None
    m = JSON_RE.search(line)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None

# ----------------------------
# Serial / stdin reader
# ----------------------------
def serial_or_stdin_lines(use_serial: bool):
    if use_serial:
        if serial is None:
            raise RuntimeError("pyserial not installed. Install: pip install pyserial")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        try:
            while True:
                line = ser.readline()
                if not line:
                    continue
                yield line.decode("utf-8", errors="replace").strip()
        finally:
            try:
                ser.close()
            except Exception:
                pass
    else:
        for line in sys.stdin:
            yield line.strip()

# ----------------------------
# Thread -> Queue
# ----------------------------
q = queue.Queue(maxsize=200)

def reader_thread(use_serial: bool):
    try:
        for line in serial_or_stdin_lines(use_serial):
            if DEBUG:
                print("LINE:", line)

            payload = extract_json_payload(line)
            if not payload:
                continue

            if DEBUG:
                print("PAYLOAD:", payload)

            # Keep only freshest data (drop old if queue full)
            try:
                q.put_nowait(payload)
            except queue.Full:
                try:
                    _ = q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    pass
    except Exception as e:
        print("Reader thread crashed:", repr(e), file=sys.stderr)

# ----------------------------
# 3D Plot
# ----------------------------
def set_limits(ax):
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)

def main():
    use_serial = True
    global SERIAL_PORT, BAUD_RATE, DEBUG

    if "--stdin" in sys.argv:
        use_serial = False
    if "--port" in sys.argv:
        i = sys.argv.index("--port")
        SERIAL_PORT = sys.argv[i + 1]
    if "--baud" in sys.argv:
        i = sys.argv.index("--baud")
        BAUD_RATE = int(sys.argv[i + 1])
    if "--debug" in sys.argv:
        DEBUG = True

    t = threading.Thread(target=reader_thread, args=(use_serial,), daemon=True)
    t.start()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    fig.canvas.manager.set_window_title("3D Azimuth+Elevation (front=120°, CCW)")

    # Unit sphere
    u = np.linspace(0, 2*np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    uu, vv = np.meshgrid(u, v)
    X = np.cos(uu) * np.sin(vv)
    Y = np.sin(uu) * np.sin(vv)
    Z = np.cos(vv)
    ax.plot_wireframe(X, Y, Z, linewidth=0.5, alpha=0.25)

    # Axes lines
    ax.plot([0, 0],   [0, 1.2], [0, 0], linewidth=2)  # forward
    ax.plot([0, 1.2], [0, 0],   [0, 0], linewidth=2)  # right
    ax.plot([0, 0],   [0, 0],   [0, 1.2], linewidth=2)  # up
    ax.text(0, 1.25, 0, "FORWARD", fontsize=10)
    ax.text(1.25, 0, 0, "RIGHT", fontsize=10)
    ax.text(0, 0, 1.25, "UP", fontsize=10)

    set_limits(ax)
    ax.set_xlabel("X (right)")
    ax.set_ylabel("Y (forward)")
    ax.set_zlabel("Z (up)")

    title = ax.set_title("Waiting for data...")

    # Initial visuals
    qv = ax.quiver(0, 0, 0, 0, 1, 0, length=1.0, normalize=True)
    pt = ax.scatter([0], [1], [0], s=40)

    # Latest known
    raw_az = None
    rel_az = None
    elev = 0.0
    dist = None
    vec = (0.0, 1.0, 0.0)

    def animate(_):
        nonlocal qv, pt, raw_az, rel_az, elev, dist, vec

        # Drain queue -> keep newest payload only
        newest = None
        while True:
            try:
                newest = q.get_nowait()
            except queue.Empty:
                break

        if newest is not None:
            try:
                raw = float(newest.get("azimuth"))
                elev = float(newest.get("elevation", 0.0))
                dist = newest.get("distance", None)

                raw_az = wrap360(raw)
                rel_az = azimuth_to_relative_deg(raw_az)
                vec = az_el_to_vector(rel_az, elev)
            except Exception:
                pass

        # Update arrow + point
        x, y, z = vec
        qv.remove()
        qv = ax.quiver(0, 0, 0, x, y, z, length=1.0, normalize=True)
        pt._offsets3d = ([x], [y], [z])

        if raw_az is None:
            title.set_text("Waiting for data...")
        else:
            title.set_text(
                f"raw az={raw_az:.1f}° | rel az(from front)={rel_az:.1f}° | elev={elev:.1f}°"
                + (f" | dist={dist}" if dist is not None else "")
            )
        return []

    anim = FuncAnimation(fig, animate, interval=33, blit=False, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    main()

