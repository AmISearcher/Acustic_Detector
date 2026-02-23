import serial, time, math

PORT="/dev/ttyUSB0"
ser = serial.Serial(PORT, 115200, timeout=0.1)
time.sleep(2)

def send(cmd: str):
    ser.write((cmd + "\n").encode("ascii"))

# set zero
send("Z")
time.sleep(0.2)

pan = 0.0
tilt = 0.0

MAX_PAN_SPEED  = 120.0  # deg/s (піднімай, якщо хочеш швидше)
MAX_TILT_SPEED = 60.0   # deg/s

PAN_MIN, PAN_MAX = -180, 180
TILT_MIN, TILT_MAX = -30, 90

last = time.time()

while True:
    now = time.time()
    dt = now - last
    last = now

    # ---- EMULATE joystick axis [-1..+1]
    # e.g. sine wave from -1..+1
    stick_x = math.sin(now * 0.6)
    stick_y = 0.0

    # deadzone like joystick
    if abs(stick_x) < 0.05: stick_x = 0.0
    if abs(stick_y) < 0.05: stick_y = 0.0

    # speed control
    pan_speed  = stick_x * MAX_PAN_SPEED
    tilt_speed = stick_y * MAX_TILT_SPEED

    # integrate to position
    pan  += pan_speed * dt
    tilt += tilt_speed * dt

    # clamp to limits
    pan  = max(PAN_MIN, min(PAN_MAX, pan))
    tilt = max(TILT_MIN, min(TILT_MAX, tilt))

    # send absolute position
    send(f"P{pan:.2f}")
    send(f"T{tilt:.2f}")

    time.sleep(0.01)  # ~100 Hz loop
