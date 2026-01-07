import serial
import serial.serialutil
# !!! ЗМІНІТЬ ЦЕ НА ПОРТ ВАШОГО АДАПТЕРА НА UBUNTU !!!
RECEIVE_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
print(f"Attempting to open {RECEIVE_PORT} @ {BAUD_RATE}")
try:
# Відкриваємо послідовний порт
    ser = serial.Serial(RECEIVE_PORT, BAUD_RATE, timeout=None) #timeout=None для блокуючого читання
    print("SERIAL PORT OPENED. Listening for data...")

    while True:
        # Читаємо рядок до символу переведення рядка ('\n')
        line = ser.readline()
        if line:
            try:
                # Декодуємо байти в UTF-8
                data = line.decode('utf-8').strip()
                print(f"RECEIVED: {data}")
            except UnicodeDecodeError:
                print("Received non-UTF-8 data.")

except serial.serialutil.SerialException as e:
    print(f"Error opening port {RECEIVE_PORT}: {e}")
except KeyboardInterrupt:
    print("\n Exiting reader.")
    if 'ser' in locals() and ser.is_open:
        ser.close()


#### RECEIVED: {"azimuth": 243, "distance": 1000, "elevation": 0, "timestamp": 1765970233780}
