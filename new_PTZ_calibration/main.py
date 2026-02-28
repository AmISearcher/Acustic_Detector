#!/usr/bin/env python3
import time
import threading
import signal
import sys

import serial

# =========================
# CONFIG
# =========================
GIMBAL_USB_PORT = "/dev/ttyUSB0"
GIMBAL_BAUD = 115200
GIMBAL_INIT_SLEEP_S = 2.0

# Safety limits (set wide first, then tighten)
PAN_LIMIT = 3000.0
TILT_LIMIT_UP = 1200.0
TILT_LIMIT_DOWN = -1200.0

# Step for keyboard nudges
PAN_STEP = 20.0
TILT_STEP = 10.0

# =========================
# STOP CONTROL
# =========================
stop_event = threading.Event()
signal.signal(signal.SIGINT, lambda *_: stop_event.set())
signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# =========================
# USB PTZ
# =========================
class UsbPTZ:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.2)
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

def print_help():
    print(
        "\nCommands:\n"
        "  set <pan> <tilt>     -> move absolute to pan/tilt\n"
        "  pan <value>          -> set pan only\n"
        "  tilt <value>         -> set tilt only\n"
        "  zero                 -> firmware zero (Z) + set software state to 0,0\n"
        "  stop                 -> stop motion (S)\n"
        "  sweep_pan <a> <b> <step>  -> sweep pan a..b (keeps current tilt)\n"
        "  sweep_tilt <a> <b> <step> -> sweep tilt a..b (keeps current pan)\n"
        "  nudge                -> interactive keyboard nudges (a/d pan, w/s tilt)\n"
        "  limits               -> print limits\n"
        "  q                    -> quit\n"
    )

def do_sweep(ptz, kind, start, end, step, pan, tilt, sleep_s=0.15):
    if step == 0:
        print("[ERR] step can't be 0")
        return pan, tilt
    if start < end and step < 0:
        step = -step
    if start > end and step > 0:
        step = -step

    v = start
    while True:
        if stop_event.is_set():
            break

        if kind == "pan":
            pan = clamp(v, -PAN_LIMIT, PAN_LIMIT)
        else:
            tilt = clamp(v, TILT_LIMIT_DOWN, TILT_LIMIT_UP)

        ptz.move_abs(pan, tilt)
        print(f"[SWEEP] pan={pan:.2f} tilt={tilt:.2f}")
        time.sleep(sleep_s)

        if (step > 0 and v >= end) or (step < 0 and v <= end):
            break
        v += step

    return pan, tilt

def nudge_mode(ptz, pan, tilt):
    """
    Keyboard nudge mode with clean terminal redraw + live values.

    Keys:
      a/d -> pan -/+
      w/s -> tilt +/-
      z   -> firmware zero + software reset to 0,0 + send P0/T0
      x   -> stop (S)
      q   -> exit nudge
    """
    import sys
    import time

    print(
        "\nNUDGE MODE (live display):\n"
        "  a/d: pan -/+ \n"
        "  w/s: tilt +/-(up/down)\n"
        "  z: zero\n"
        "  x: stop\n"
        "  q: quit nudge mode\n"
    )

    def redraw(status=""):
        # Clear screen + move cursor to top-left
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write("NUDGE MODE (live)\n")
        sys.stdout.write("-----------------\n")
        sys.stdout.write(f"PAN : {pan:10.2f}\n")
        sys.stdout.write(f"TILT: {tilt:10.2f}\n")
        sys.stdout.write("\nKeys: a/d pan, w/s tilt, z zero, x stop, q quit\n")
        if status:
            sys.stdout.write(f"\n{status}\n")
        sys.stdout.flush()

    redraw("Ready")

    try:
        import termios, tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setraw(fd)

        while not stop_event.is_set():
            ch = sys.stdin.read(1)

            if ch == "q":
                redraw("Exit nudge")
                break

            elif ch == "a":
                pan = clamp(pan - PAN_STEP, -PAN_LIMIT, PAN_LIMIT)
                ptz.move_abs(pan, tilt)
                redraw("Moved")

            elif ch == "d":
                pan = clamp(pan + PAN_STEP, -PAN_LIMIT, PAN_LIMIT)
                ptz.move_abs(pan, tilt)
                redraw("Moved")

            elif ch == "w":
                tilt = clamp(tilt + TILT_STEP, TILT_LIMIT_DOWN, TILT_LIMIT_UP)
                ptz.move_abs(pan, tilt)
                redraw("Moved")

            elif ch == "s":
                tilt = clamp(tilt - TILT_STEP, TILT_LIMIT_DOWN, TILT_LIMIT_UP)
                ptz.move_abs(pan, tilt)
                redraw("Moved")

            elif ch == "z":
                # Firmware zero + also reset our software state
                ptz.set_zero()
                time.sleep(0.2)
                pan, tilt = 0.0, 0.0
                ptz.move_abs(pan, tilt)
                redraw("Zeroed (firmware + software)")

            elif ch == "x":
                ptz.stop()
                redraw("STOP (S sent)")

            else:
                redraw(f"Unknown key: {repr(ch)}")

    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass
        print("\n[NUDGE] exit")
    return pan, tilt

def main():
    ptz = UsbPTZ(GIMBAL_USB_PORT, GIMBAL_BAUD)
    print("[INFO] PTZ ID:", ptz.identify())
    pan = 0.0
    tilt = 0.0

    print_help()
    try:
        while not stop_event.is_set():
            cmd = input("\nptz> ").strip()
            if not cmd:
                continue

            parts = cmd.split()
            op = parts[0].lower()

            if op in ("q", "quit", "exit"):
                break

            elif op == "help":
                print_help()

            elif op == "limits":
                print(f"PAN_LIMIT=±{PAN_LIMIT}, TILT=[{TILT_LIMIT_DOWN}, {TILT_LIMIT_UP}]")
                print(f"PAN_STEP={PAN_STEP}, TILT_STEP={TILT_STEP}")

            elif op == "set" and len(parts) == 3:
                pan = clamp(float(parts[1]), -PAN_LIMIT, PAN_LIMIT)
                tilt = clamp(float(parts[2]), TILT_LIMIT_DOWN, TILT_LIMIT_UP)
                ptz.move_abs(pan, tilt)
                print(f"[MOVE] pan={pan:.2f} tilt={tilt:.2f}")

            elif op == "pan" and len(parts) == 2:
                pan = clamp(float(parts[1]), -PAN_LIMIT, PAN_LIMIT)
                ptz.move_abs(pan, tilt)
                print(f"[MOVE] pan={pan:.2f} tilt={tilt:.2f}")

            elif op == "tilt" and len(parts) == 2:
                tilt = clamp(float(parts[1]), TILT_LIMIT_DOWN, TILT_LIMIT_UP)
                ptz.move_abs(pan, tilt)
                print(f"[MOVE] pan={pan:.2f} tilt={tilt:.2f}")

            elif op == "zero":
                ptz.set_zero()
                time.sleep(0.2)
                pan, tilt = 0.0, 0.0
                ptz.move_abs(pan, tilt)
                print("[ZERO] firmware + software state reset to pan=0 tilt=0")

            elif op == "stop":
                ptz.stop()
                print("[STOP]")

            elif op == "sweep_pan" and len(parts) == 4:
                a = float(parts[1]); b = float(parts[2]); step = float(parts[3])
                pan, tilt = do_sweep(ptz, "pan", a, b, step, pan, tilt)

            elif op == "sweep_tilt" and len(parts) == 4:
                a = float(parts[1]); b = float(parts[2]); step = float(parts[3])
                pan, tilt = do_sweep(ptz, "tilt", a, b, step, pan, tilt)

            elif op == "nudge":
                pan, tilt = nudge_mode(ptz, pan, tilt)

            else:
                print("[ERR] unknown command or wrong args. type 'help'.")

    finally:
        try:
            ptz.stop()
        except Exception:
            pass
        try:
            ptz.close()
        except Exception:
            pass
        print("\n[INFO] closed")

if __name__ == "__main__":
    main()