
import serial
import time
import threading

laser_distance = None

def get_laser_distance():
        return laser_distance

def start_laser_reader():
    def laser_reader():
        global laser_distance
        port = "/dev/ttyUSB0"
        baud = 9600

        START_CMD = b"\xAE\xA7\x04\x00\x0E\x12\xBC\xBE"
        STOP_CMD  = b"\xAE\xA7\x04\x00\x0F\x13\xBC\xBE"
        RED_DOT_ON = b"\xAE\xA7\x05\x00\x40\x01\x46\xBC\xBE"

        try:
            ser = serial.Serial(port, baud, bytesize=serial.EIGHTBITS,
                                parity=serial.PARITY_NONE,
                                stopbits=serial.STOPBITS_ONE,
                                timeout=1)
            ser.reset_input_buffer()
            ser.write(RED_DOT_ON)
            ser.flush()
            time.sleep(0.1)
            print("[INFO] Red dot laser activated.")
            ser.write(START_CMD)
            ser.flush()
            print("Started continuous measurement...")
        except Exception as e:
            print(f"[ERROR] Failed to open serial port: {e}")
            return

        buffer = bytearray()
        while True:
            try:
                chunk = ser.read(1024)
                if not chunk:
                    continue
                buffer += chunk

                while True:
                    start_idx = buffer.find(b"\xAE\xA7")
                    if start_idx == -1:
                        buffer = bytearray([0xAE]) if buffer and buffer[-1] == 0xAE else bytearray()
                        break

                    if start_idx > 0:
                        del buffer[:start_idx]

                    if len(buffer) < 5:
                        break

                    length_byte = buffer[2]
                    total_frame_len = length_byte + 4
                    if len(buffer) < total_frame_len:
                        break

                    frame = buffer[:total_frame_len]
                    if frame[-2:] != b"\xBC\xBE":
                        del buffer[0]
                        continue

                    command = frame[4]
                    if command != 0x85:
                        del buffer[:total_frame_len]
                        continue

                    if len(frame) < 24:
                        del buffer[:total_frame_len]
                        continue

                    high_byte = frame[7]
                    low_byte = frame[8]
                    raw_distance = (high_byte << 8) | low_byte
                    signed_distance = raw_distance - 0x10000 if raw_distance & 0x8000 else raw_distance

                    unit_code = frame[23]
                    if unit_code == 0x01:
                        distance_m = signed_distance * 0.1
                    elif unit_code == 0x02:
                        distance_m = signed_distance * 0.1 * 0.9144
                    elif unit_code == 0x03:
                        distance_m = signed_distance * 0.1 * 0.3048
                    else:
                        del buffer[:total_frame_len]
                        continue

                    laser_distance = distance_m
                    del buffer[:total_frame_len]

            except Exception as e:
                print(f"[ERROR] Laser read failed: {e}")
                break

    thread = threading.Thread(target=laser_reader, daemon=True)
    thread.start()