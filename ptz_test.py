#!/usr/bin/env python3
import serial
import time

# =========================
# CONFIG
# =========================
PORT = "/dev/ttyUSB0"   # або /dev/ttyACM0
BAUD = 115200

PAN_MIN = -180
PAN_MAX = 180

SEND_HZ = 50.0          # не більше 50 команд/сек
MOVE_DELAY = 0.05       # пауза між великими цілями

# =========================

def send(ser, cmd: str):
    ser.write((cmd + "\n").encode("ascii"))

def main():
    print("[INFO] Opening port...")
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    time.sleep(2.0)  # MCU reset

    print("[INFO] Identify:")
    send(ser, "I")
    print(ser.readline().decode(errors="replace").strip())

    print("[INFO] Set ZERO")
    send(ser, "Z")
    print(ser.readline().decode(errors="replace").strip())

    time.sleep(0.3)

    print("[INFO] TURBO SPIN MODE STARTED")
    print("Press Ctrl+C to stop")

    last_send = 0.0
    send_dt = 1.0 / SEND_HZ

    direction = 1  # 1 -> to max, -1 -> to min

    try:
        while True:
            now = time.time()

            if (now - last_send) >= send_dt:
                if direction > 0:
                    target = PAN_MAX
                else:
                    target = PAN_MIN

                send(ser, f"P{target}")
                last_send = now

                # переключаємо напрямок
                direction *= -1

            time.sleep(MOVE_DELAY)

    except KeyboardInterrupt:
        print("\n[INFO] STOP")
        send(ser, "S")
        ser.close()
        print("[INFO] Done.")

if __name__ == "__main__":
    main()
