import serial
import time
try:
    with serial.Serial('COM3', 921600, timeout=1) as ser:
        print("Reading COM3...")
        start = time.time()
        while time.time() - start < 3:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"[COM3] {line}")
except Exception as e:
    print(e)
