import serial
import time

PORT = "/dev/ttyUSB0"   # або /dev/ttyACM0
BAUD = 115200

def send(ser, cmd: str):
    ser.write((cmd + "\n").encode("ascii"))

def read_line(ser) -> str:
    raw = ser.readline()
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace").strip()

def query_pos(ser):
    send(ser, "?")
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

def move_and_report(ser, pan, tilt=None, wait=1.5):
    if pan is not None:
        send(ser, f"P{pan:.2f}")
    if tilt is not None:
        send(ser, f"T{tilt:.2f}")
    time.sleep(wait)
    p, t, raw = query_pos(ser)
    print(f"After move P={pan} T={tilt} -> {raw}")

print("[INFO] Opening port...")
ser = serial.Serial(PORT, BAUD, timeout=0.2)
time.sleep(2.0)  # MCU reset

send(ser, "I")
print("[ID ]", read_line(ser))

send(ser, "Z")
print("[ZERO]", read_line(ser))

# read initial pos
p0, t0, raw0 = query_pos(ser)
print("[POS0]", raw0)

# Try a few targets
for target in [10, 30, 60, 90, 120, -10, -30, -60, -90]:
    print(f"\n[TEST] P{target}")
    move_and_report(ser, pan=target, tilt=None, wait=1.5)

print("\n[STOP]")
send(ser, "S")
print(read_line(ser))

ser.close()
print("[INFO] Done.")
