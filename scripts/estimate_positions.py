import sys
import time
import numpy as np
sys.path.insert(0, "scripts")
from parse_csi import CSIDataReader, extract_amplitude_phase
import threading
import collections

# We will collect data for 5 seconds
DURATION = 5

data = {
    "COM3": collections.defaultdict(list),
    "COM5": collections.defaultdict(list),
}

rssi_data = {
    "COM3": collections.defaultdict(list),
    "COM5": collections.defaultdict(list),
}

def collect_data(port):
    print(f"[{port}] Starting collection...")
    try:
        with CSIDataReader(port=port, baud_rate=921600) as reader:
            for pkt in reader.read_stream(duration=DURATION):
                mac = pkt.get('mac', 'unknown')
                amp, _ = extract_amplitude_phase(pkt["raw_data"])
                # Also extract RSSI from the raw packet if available, but for now we use mean amplitude
                rssi = pkt.get('rssi', 0) 
                
                data[port][mac].append(np.mean(amp))
                rssi_data[port][mac].append(rssi)
                
    except Exception as e:
        print(f"Error on {port}: {e}")

# Run threads
t3 = threading.Thread(target=collect_data, args=("COM3",))
t5 = threading.Thread(target=collect_data, args=("COM5",))

t3.start()
t5.start()
t3.join()
t5.join()

print("\n--- Position Estimation Analysis ---")
macs = sorted(list(set(data["COM3"].keys()) | set(data["COM5"].keys())))

for mac in macs:
    amp3 = np.mean(data["COM3"].get(mac, [0]))
    amp5 = np.mean(data["COM5"].get(mac, [0]))
    
    rssi3 = np.mean(rssi_data["COM3"].get(mac, [-100]))
    rssi5 = np.mean(rssi_data["COM5"].get(mac, [-100]))
    
    node_id = mac[-2:]
    
    print(f"\nNode {node_id} ({mac}):")
    print(f"  COM3 (Right) -> Mean Amp: {amp3:.2f}, RSSI: {rssi3:.2f}")
    print(f"  COM5 (Left)  -> Mean Amp: {amp5:.2f}, RSSI: {rssi5:.2f}")
    
    # Simple logic
    if amp3 > amp5 * 1.5:
        pos_x = "Very Close to Right (COM3)"
    elif amp3 > amp5 * 1.1:
        pos_x = "Slightly Right"
    elif amp5 > amp3 * 1.5:
        pos_x = "Very Close to Left (COM5)"
    elif amp5 > amp3 * 1.1:
        pos_x = "Slightly Left"
    else:
        pos_x = "Centered (Equal distance to Left and Right)"
        
    print(f"  => Estimated X-Axis: {pos_x}")
    
    # Distance estimation based on absolute amplitude
    max_amp = max(amp3, amp5)
    if max_amp > 40:
        pos_y = "Near to the receivers"
    elif max_amp > 20:
        pos_y = "Mid distance"
    else:
        pos_y = "Far away / Behind walls"
        
    print(f"  => Estimated Y-Axis: {pos_y}")
