import serial
import time
import sys

PORT = "COM7"
BAUD = 115200

try:
    print(f"Connecting to ESP8266 on {PORT}...")
    with serial.Serial(PORT, BAUD, timeout=1) as ser:
        print("Connected. Listening for 5 seconds...")
        start = time.time()
        while time.time() - start < 5:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"[ESP8266] {line}")
except Exception as e:
    print(f"Error: {e}")
