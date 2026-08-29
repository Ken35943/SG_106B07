import sys
import time
sys.path.insert(0, "scripts")
from parse_csi import CSIDataReader

PORT = "COM3"
BAUD = 921600
DURATION = 5  # Check for 5 seconds

print(f"Listening on {PORT} for ESP8266 beacons...")
mac_counts = {}

try:
    with CSIDataReader(port=PORT, baud_rate=BAUD) as reader:
        start = time.monotonic()
        for pkt in reader.read_stream(duration=DURATION):
            mac = pkt.get('mac', 'unknown')
            if mac not in mac_counts:
                mac_counts[mac] = 0
            mac_counts[mac] += 1
            
            # Print the first time we see a new MAC
            if mac_counts[mac] == 1:
                print(f"[*] Discovered new MAC: {mac}")
                
except Exception as e:
    print(f"Error: {e}")

print("\n--- Summary ---")
for mac, count in mac_counts.items():
    print(f"MAC: {mac} | Packets: {count}")
    
if not mac_counts:
    print("No packets received.")
