#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAST PTZ test driver (USB Virtual COM) for API:
  P<float>\n, T<float>\n, Z\n, S\n, ?\n, I\n

Goal: make gimbal move as fast as possible from the PC side, but with smooth
acceleration/braking when changing direction (joystick-like).

Two modes:
  MODE = "velocity"  -> integrate deg/s into absolute P/T with accel limiting (smooth)
  MODE = "bounce"    -> alternate absolute targets between limits (forces accel)

Run:
  python3 fast_ptz.py
Stop:
  Ctrl+C  (sends S\n)
"""

import time
import signal
import serial

# =========================
# CONFIG (EDIT THESE)
# =========================
PORT = "/dev/ttyUSB0"     # default; may be /dev/ttyACM0
BAUD = 115200
SER_TIMEOUT = 0.05        # read timeout
WRITE_TIMEOUT = 0.2

ZERO_ON_START = True      # send Z\n after connect
PRINT_ID = True           # send I\n and print response

# Choose: "velocity" or "bounce"
MODE = "velocity"
# MODE = "bounce"

# Mechanical/firmware limits (set SAFE values!)
# NOTE: Your firmware may clamp internally (e.g. +/-180). Larger values won't increase real range.
PAN_MIN = -2880.0
PAN_MAX =  2880.0
TILT_MIN = -30.0
TILT_MAX =  90.0

# Command sending behavior
PTZ_SEND_HZ = 60.0        # how often we send P/T (30–80 Hz typical)
MIN_DEG_STEP = 0.5        # do NOT send micro-updates smaller than this (deg)
SEND_TILT_TOO = False     # for pure fast pan test, keep tilt fixed

# Optional position querying (slows overall throughput; keep OFF for max speed)
QUERY_POS_EVERY_S = 0.0   # 0.0 disables; e.g. 0.5 queries twice per second

# =========================
# "velocity" mode params
# =========================
MAX_PAN_SPEED_DEGPS  = 570.0   # desired max deg/s (firmware caps anyway)
MAX_TILT_SPEED_DEGPS = 360.0

# Smooth accel/braking (deg/s^2). Higher = snappier, lower = smoother.
MAX_PAN_ACCEL_DEGPS2  = 1200.0
MAX_TILT_ACCEL_DEGPS2 = 800.0

# constant "stick" values in [-1..+1]; set sign to choose direction
STICK_X = +1.0   # +1.0 = pan right; -1.0 = pan left; 0.0 = stop pan
STICK_Y =  0.0   # +1.0 tilt up, -1.0 tilt down

# If you want auto-bounce in velocity mode when hitting limits:
VELOCITY_AUTO_BOUNCE = True   # reverses direction at limits

# =========================
# "bounce" mode params
# =========================
BOUNCE_PAN_A = PAN_MIN
BOUNCE_PAN_B = PAN_MAX
BOUNCE_TILT  = 0.0

USE_QUERY_FOR_BOUNCE = False
NEAR_TARGET_DEG = 3.0
BOUNCE_SLICE_S = 1.0

# =========================
# END CONFIG
# =========================

stop_flag = False

def on_sigint(sig, frame):
    global stop_flag
    stop_flag = True

signal.signal(signal.SIGINT, on_sigint)
signal.signal(signal.SIGTERM, on_sigint)

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def send_line(ser: serial.Serial, line: str):
    ser.write((line + "\n").encode("ascii", errors="ignore"))

def read_line(ser: serial.Serial) -> str:
    raw = ser.readline()
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace").strip()

def query_pos(ser: serial.Serial):
    send_line(ser, "?")
    line = read_line(ser)
    # expected: POS:90.50,-15.00
    if line.startswith("POS:"):
        try:
            payload = line.split("POS:", 1)[1]
            a, b = payload.split(",", 1)
            return float(a), float(b), line
        except Exception:
            return None, None, line
    return None, None, line

def accel_limit(current_speed: float, target_speed: float, max_accel: float, dt: float) -> float:
    """
    Limit rate of change of speed: |d(speed)/dt| <= max_accel
    """
    if dt <= 0:
        return current_speed
    diff = target_speed - current_speed
    max_step = max_accel * dt
    if diff > max_step:
        diff = max_step
    elif diff < -max_step:
        diff = -max_step
    return current_speed + diff

def main():
    print(f"[INFO] Opening {PORT} @ {BAUD} ...")
    ser = serial.Serial(
        PORT,
        BAUD,
        timeout=SER_TIMEOUT,
        write_timeout=WRITE_TIMEOUT
    )

    # Many MCU boards reset on serial open:
    time.sleep(2.0)

    # Flush any boot text
    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception:
        pass

    if PRINT_ID:
        send_line(ser, "I")
        time.sleep(0.05)
        print("[ID ]", read_line(ser) or "(no response)")

    if ZERO_ON_START:
        send_line(ser, "Z")
        time.sleep(0.05)
        resp = read_line(ser)
        if resp:
            print("[ZERO]", resp)
        else:
            print("[ZERO] (no response)")

    # Start at current defined zero
    pan = 0.0
    tilt = 0.0
    pan = clamp(pan, PAN_MIN, PAN_MAX)
    tilt = clamp(tilt, TILT_MIN, TILT_MAX)

    # Velocity state (smoothed)
    current_pan_speed = 0.0
    current_tilt_speed = 0.0

    # Sending loop timing
    send_dt = 1.0 / max(1e-6, PTZ_SEND_HZ)
    last_send = 0.0
    last_query = 0.0

    last_sent_pan = pan
    last_sent_tilt = tilt

    # For bounce mode state
    bounce_target_pan = BOUNCE_PAN_B
    bounce_target_tilt = clamp(BOUNCE_TILT, TILT_MIN, TILT_MAX)
    bounce_last_switch = time.time()

    # For velocity mode direction handling
    stick_x = float(STICK_X)
    stick_y = float(STICK_Y)

    print(f"[INFO] MODE={MODE}")
    print("[INFO] Ctrl+C to STOP")

    t_prev = time.time()

    try:
        while not stop_flag:
            now = time.time()
            dt = now - t_prev
            t_prev = now

            # -------------------------
            # MODE: velocity integration (smooth accel/brake)
            # -------------------------
            if MODE.lower() == "velocity":
                # target speeds from "stick"
                target_pan_speed = stick_x * MAX_PAN_SPEED_DEGPS
                target_tilt_speed = stick_y * MAX_TILT_SPEED_DEGPS

                # accel/brake limiting (smooth direction changes)
                current_pan_speed = accel_limit(current_pan_speed, target_pan_speed, MAX_PAN_ACCEL_DEGPS2, dt)
                current_tilt_speed = accel_limit(current_tilt_speed, target_tilt_speed, MAX_TILT_ACCEL_DEGPS2, dt)

                # integrate speed -> absolute setpoint
                pan += current_pan_speed * dt
                tilt += current_tilt_speed * dt

                # limit handling (software limits)
                if VELOCITY_AUTO_BOUNCE:
                    # PAN bounce
                    if pan >= PAN_MAX:
                        pan = PAN_MAX
                        # flip stick direction (target), but speed will ramp smoothly due to accel_limit
                        stick_x = -abs(stick_x) if stick_x != 0 else -1.0
                    elif pan <= PAN_MIN:
                        pan = PAN_MIN
                        stick_x = +abs(stick_x) if stick_x != 0 else +1.0

                    # TILT bounce
                    if tilt >= TILT_MAX:
                        tilt = TILT_MAX
                        stick_y = -abs(stick_y) if stick_y != 0 else -1.0
                    elif tilt <= TILT_MIN:
                        tilt = TILT_MIN
                        stick_y = +abs(stick_y) if stick_y != 0 else +1.0
                else:
                    pan = clamp(pan, PAN_MIN, PAN_MAX)
                    tilt = clamp(tilt, TILT_MIN, TILT_MAX)

            # -------------------------
            # MODE: aggressive bounce
            # -------------------------
            elif MODE.lower() == "bounce":
                # Choose target
                target_pan = clamp(bounce_target_pan, PAN_MIN, PAN_MAX)
                target_tilt = clamp(bounce_target_tilt, TILT_MIN, TILT_MAX)

                # Update setpoint directly (absolute)
                pan = target_pan
                tilt = target_tilt

                if USE_QUERY_FOR_BOUNCE:
                    if QUERY_POS_EVERY_S > 0 and (now - last_query) >= QUERY_POS_EVERY_S:
                        last_query = now
                        p, t, raw = query_pos(ser)
                        if p is not None:
                            if abs(p - target_pan) <= NEAR_TARGET_DEG:
                                bounce_target_pan = BOUNCE_PAN_A if bounce_target_pan == BOUNCE_PAN_B else BOUNCE_PAN_B
                else:
                    if (now - bounce_last_switch) >= BOUNCE_SLICE_S:
                        bounce_last_switch = now
                        bounce_target_pan = BOUNCE_PAN_A if bounce_target_pan == BOUNCE_PAN_B else BOUNCE_PAN_B

            else:
                print(f"[ERR] Unknown MODE: {MODE}")
                break

            # -------------------------
            # Rate-limited sending + MIN_DEG_STEP
            # -------------------------
            if (now - last_send) >= send_dt:
                do_send_pan = abs(pan - last_sent_pan) >= MIN_DEG_STEP
                do_send_tilt = SEND_TILT_TOO and (abs(tilt - last_sent_tilt) >= MIN_DEG_STEP)

                if do_send_pan:
                    send_line(ser, f"P{pan:.2f}")
                    last_sent_pan = pan

                if do_send_tilt:
                    send_line(ser, f"T{tilt:.2f}")
                    last_sent_tilt = tilt

                last_send = now

            # Optional, low-rate info query (can reduce speed—use sparingly)
            if QUERY_POS_EVERY_S > 0 and (now - last_query) >= QUERY_POS_EVERY_S:
                last_query = now
                p, t, raw = query_pos(ser)
                if raw:
                    print("[POS]", raw)

            time.sleep(0.001)

    finally:
        # Emergency STOP
        try:
            send_line(ser, "S")
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass
        print("\n[INFO] Stopped (S sent) and port closed.")

if __name__ == "__main__":
    main()
