import serial
import time

PORT = "/dev/ttyUSB0"   # якщо треба — заміни на /dev/ttyACM0
BAUD = 115200

def read_line(ser):
    raw = ser.readline()
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace").strip()

def send(ser, cmd):
    ser.write((cmd + "\n").encode("ascii"))

print("[INFO] Opening port...")
ser = serial.Serial(PORT, BAUD, timeout=0.2)
time.sleep(2.0)  # MCU reset

print("[INFO] Identify:")
send(ser, "I")
print("->", read_line(ser))

print("[INFO] Set ZERO")
send(ser, "Z")
print("->", read_line(ser))

time.sleep(0.5)

# =========================
# TEST 1: 0 → 90 degrees
# =========================
print("\n[TEST] Moving to +90°")
start = time.time()
send(ser, "P90")

# чекаємо поки приблизно доїде
time.sleep(3)

elapsed = time.time() - start
print(f"Time approx to 90°: {elapsed:.2f} seconds")

# =========================
# TEST 2: 90 → -90 degrees
# =========================
print("\n[TEST] Moving to -90°")
start = time.time()
send(ser, "P-90")

time.sleep(3)

elapsed = time.time() - start
print(f"Time approx to -90°: {elapsed:.2f} seconds")

print("\n[INFO] Emergency stop")
send(ser, "S")

ser.close()
print("[INFO] Done.")
