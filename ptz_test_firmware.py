import serial, time

ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.2)
time.sleep(2)

def send(cmd):
    ser.write((cmd+"\n").encode())

def read():
    r = ser.readline()
    return r.decode(errors="replace").strip() if r else ""

send("Z")
time.sleep(0.2)

for target in [10, 30, 60, 90, 120, 150, 180]:
    send(f"P{target}")
    time.sleep(1)
    send("?")
    print("Target:", target, "->", read())
