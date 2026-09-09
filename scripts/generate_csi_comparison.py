"""Generate real CSI comparison heatmap: Real Human Fall Forward vs. Normal Walking."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_and_preprocess(filepath, start_frame=0, num_frames=500):
    df = pd.read_csv(filepath)
    amp_cols = [c for c in df.columns if c.startswith('amplitude_')]
    data = df[amp_cols].values
    
    # Extract window of interest
    end_frame = min(start_frame + num_frames, len(data))
    data = data[start_frame:end_frame, :]
    if len(data) < num_frames:
        # pad if needed
        pad = np.repeat(data[-1:, :], num_frames - len(data), axis=0)
        data = np.vstack([data, pad])
    
    # Convert to log-amplitude (dB) relative to median per subcarrier
    # to highlight dynamic perturbations
    median = np.median(data, axis=0, keepdims=True)
    median = np.where(median <= 0, 1.0, median)
    data_safe = np.where(data <= 0, 1.0, data)
    diff_db = 20.0 * np.log10(data_safe / median)
    
    # Clip extreme outlier values for clean visualization
    diff_db = np.clip(diff_db, -15.0, 15.0)
    return diff_db

def generate_comparison_plot():
    # Real Fall Forward: sample_20260909_200400.csv (Peak impact is at frame 395)
    # Taking frames 145 to 645 (500 frames = 5 seconds at 100 Hz)
    fall_data = load_and_preprocess(
        'data/raw/fall_forward/sample_20260909_200400.csv',
        start_frame=145,
        num_frames=500
    )
    
    # Real Walking: sample_20260715_192540.csv (first 500 frames)
    walk_data = load_and_preprocess(
        'data/raw/walking/sample_20260715_192540.csv',
        start_frame=0,
        num_frames=500
    )
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Fall
    im1 = ax1.imshow(fall_data.T, aspect='auto', cmap='viridis', vmin=-15, vmax=15, origin='upper')
    ax1.set_title('Real Human Fall Forward (Safety Mat Capture)', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Time (Frames @ 100 Hz / ~10 ms)', fontsize=11)
    ax1.set_ylabel('Subcarrier Index (HT40)', fontsize=11)
    # Add phase annotations
    ax1.axvline(x=250, color='#FF1744', linestyle='--', alpha=0.8, lw=1.5)
    ax1.text(250, 15, ' Impact Transient', color='#FF1744', fontsize=10, fontweight='bold')
    ax1.text(60, 175, 'Pre-Fall (Still)', color='#00E5FF', fontsize=10)
    ax1.text(340, 175, 'Post-Fall (On Mat)', color='#00E676', fontsize=10)
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label('Amplitude Perturbation (dB)', fontsize=10)
    
    # Plot Walk
    im2 = ax2.imshow(walk_data.T, aspect='auto', cmap='viridis', vmin=-15, vmax=15, origin='upper')
    ax2.set_title('Real Normal Walking (Continuous Movement)', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Time (Frames @ 100 Hz / ~10 ms)', fontsize=11)
    ax2.set_ylabel('Subcarrier Index (HT40)', fontsize=11)
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label('Amplitude Perturbation (dB)', fontsize=10)
    
    plt.suptitle('WiFi CSI Amplitude Perturbation Signatures (Dual ESP32-S3 HT40)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    output_path = 'assets/csi_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated real CSI comparison at {output_path}")

if __name__ == '__main__':
    generate_comparison_plot()
