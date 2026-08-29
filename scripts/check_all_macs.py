import sys
import time
sys.path.insert(0, "scripts")
from parse_csi import CSIDataReader

PORT = "COM3"
BAUD = 921600
DURATION = 8  # Listen for 8 seconds

print(f"Listening on {PORT} for ESP8266 beacons...")
mac_counts = {}

try:
    with CSIDataReader(port=PORT, baud_rate=BAUD) as reader:
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

print("\n--- CSI Data Reception Summary ---")
# Sort by MAC ending
sorted_macs = sorted(mac_counts.items(), key=lambda x: x[0])
for mac, count in sorted_macs:
    print(f"Node: {mac} | Received Packets: {count} | Avg Hz: {count/DURATION:.1f} packets/sec")
    
if not mac_counts:
    print("No packets received. Are the ESP8266 units powered on?")
