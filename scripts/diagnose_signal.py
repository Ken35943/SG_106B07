#!/usr/bin/env python3
"""
Quick signal quality diagnostic.
Reads 10 seconds of CSI data and compares against previous baseline.
"""
import sys
import time
import numpy as np
sys.path.insert(0, "scripts")
from parse_csi import CSIDataReader, extract_amplitude_phase

PORT = "COM3"
BAUD = 921600
DURATION = 10  # seconds

print("=" * 60)
print("  ESP32-S3 CSI Signal Quality Diagnostic")
print("  Setup: Both ESPs with 6dBi antennas")
print("=" * 60)
print(f"\n  Connecting to {PORT} @ {BAUD} baud...")

amplitudes = []
rssi_values = []
packet_times = []
subcarrier_counts = []

try:
    with CSIDataReader(port=PORT, baud_rate=BAUD) as reader:
        print(f"  Connected! Recording {DURATION} seconds...\n")
        start = time.monotonic()
        
        for pkt in reader.read_stream(duration=DURATION):
            amp, phase = extract_amplitude_phase(pkt["raw_data"])
            amplitudes.append(amp)
            subcarrier_counts.append(len(amp))
            packet_times.append(time.monotonic() - start)
            try:
                rssi_values.append(int(pkt.get("rssi", 0)))
            except (ValueError, TypeError):
                pass

except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

elapsed = time.monotonic() - start
pkt_count = len(amplitudes)

if pkt_count == 0:
    print("  No packets received! Check connections.")
    sys.exit(1)

# Analyze
most_common_nsub = max(set(subcarrier_counts), key=subcarrier_counts.count)
valid_amps = [a for a in amplitudes if len(a) == most_common_nsub]
amp_matrix = np.array(valid_amps)

rssi_arr = np.array(rssi_values)
pkt_rate = pkt_count / elapsed

# Variance per subcarrier (indicator of sensitivity)
variance_per_sub = amp_matrix.var(axis=0)
top10_var_idx = np.argsort(variance_per_sub)[-10:][::-1]

# Compare with previous data (from sitting_down sample 1)
print("=" * 60)
print("  CURRENT SESSION (Both 6dBi)")
print("=" * 60)
print(f"  Duration:          {elapsed:.1f} seconds")
print(f"  Packets received:  {pkt_count}")
print(f"  Packet rate:       {pkt_rate:.1f} pkt/s")
print(f"  Subcarriers:       {most_common_nsub}")
print(f"  Valid packets:     {len(valid_amps)} ({len(valid_amps)/pkt_count*100:.1f}%)")
print(f"")
print(f"  RSSI:")
print(f"    Mean:            {rssi_arr.mean():.1f} dBm")
print(f"    Min:             {rssi_arr.min()} dBm")
print(f"    Max:             {rssi_arr.max()} dBm")
print(f"    Std:             {rssi_arr.std():.2f}")
print(f"")
print(f"  Amplitude (all subcarriers):")
print(f"    Mean:            {amp_matrix.mean():.2f}")
print(f"    Std:             {amp_matrix.std():.2f}")
print(f"    Max:             {amp_matrix.max():.2f}")
print(f"")
print(f"  Signal Variance (sensitivity indicator):")
print(f"    Mean variance:   {variance_per_sub.mean():.2f}")
print(f"    Max variance:    {variance_per_sub.max():.2f}")
print(f"    Top-10 active subcarriers: {top10_var_idx.tolist()}")

# Load previous data for comparison
print(f"\n{'=' * 60}")
print(f"  COMPARISON WITH PREVIOUS DATA (TX: PCB antenna)")
print(f"{'=' * 60}")

import csv
from pathlib import Path

prev_file = Path("data/raw/sitting_down/sample_20260715_191838.csv")
if prev_file.exists():
    with open(prev_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    
    amp_cols = [i for i, h in enumerate(header) if h.startswith("amplitude_")]
    prev_data = np.array(rows, dtype=np.float64)
    prev_amps = prev_data[:, amp_cols]
    
    rssi_col = header.index("rssi")
    prev_rssi = prev_data[:, rssi_col]
    
    prev_var = prev_amps.var(axis=0)
    prev_rate = len(rows) / 10.0  # was 10s recording
    
    print(f"  {'Metric':<25} {'Before (PCB TX)':>15} {'Now (6dBi TX)':>15} {'Change':>10}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*10}")
    print(f"  {'Packet Rate (pkt/s)':<25} {prev_rate:>15.1f} {pkt_rate:>15.1f} {(pkt_rate/prev_rate - 1)*100:>+9.1f}%")
    print(f"  {'RSSI Mean (dBm)':<25} {prev_rssi.mean():>15.1f} {rssi_arr.mean():>15.1f} {rssi_arr.mean() - prev_rssi.mean():>+9.1f}")
    print(f"  {'Amplitude Mean':<25} {prev_amps.mean():>15.2f} {amp_matrix.mean():>15.2f} {(amp_matrix.mean()/prev_amps.mean() - 1)*100:>+9.1f}%")
    print(f"  {'Amplitude Std':<25} {prev_amps.std():>15.2f} {amp_matrix.std():>15.2f} {(amp_matrix.std()/prev_amps.std() - 1)*100:>+9.1f}%")
    print(f"  {'Mean Variance':<25} {prev_var.mean():>15.2f} {variance_per_sub.mean():>15.2f} {(variance_per_sub.mean()/prev_var.mean() - 1)*100:>+9.1f}%")
    
    # Verdict
    rssi_improved = rssi_arr.mean() > prev_rssi.mean()
    amp_improved = amp_matrix.mean() > prev_amps.mean()
    var_improved = variance_per_sub.mean() > prev_var.mean()
    
    improvements = sum([rssi_improved, amp_improved, var_improved])
    
    print(f"\n  VERDICT:")
    if rssi_improved:
        print(f"    [+] RSSI improved by {rssi_arr.mean() - prev_rssi.mean():.1f} dB (stronger signal)")
    else:
        print(f"    [-] RSSI decreased by {prev_rssi.mean() - rssi_arr.mean():.1f} dB")
    
    if amp_improved:
        print(f"    [+] Amplitude increased by {(amp_matrix.mean()/prev_amps.mean() - 1)*100:.1f}%")
    else:
        print(f"    [-] Amplitude decreased by {(1 - amp_matrix.mean()/prev_amps.mean())*100:.1f}%")
    
    if var_improved:
        print(f"    [+] Sensitivity improved by {(variance_per_sub.mean()/prev_var.mean() - 1)*100:.1f}% (more responsive to movement)")
    else:
        print(f"    [-] Sensitivity decreased by {(1 - variance_per_sub.mean()/prev_var.mean())*100:.1f}%")
    
    print()
    if improvements >= 2:
        print(f"    >>> UPGRADE TO DUAL 6dBi IS AN IMPROVEMENT! <<<")
    elif improvements == 1:
        print(f"    >>> MARGINAL CHANGE - roughly equivalent to before <<<")
    else:
        print(f"    >>> NO IMPROVEMENT - check antenna connections <<<")

else:
    print(f"  Previous data file not found for comparison.")
    print(f"  But current signal stats are shown above.")

print(f"\n{'=' * 60}")
