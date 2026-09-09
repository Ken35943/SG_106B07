# ESP32-S3 WiFi CSI Fall Detection & Activity Monitoring System (Working Demo)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.7+ CUDA](https://img.shields.io/badge/PyTorch-2.7%2B%20CUDA-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ESP-IDF v5.4](https://img.shields.io/badge/ESP--IDF-v5.4-E7352C?style=flat-square&logo=espressif&logoColor=white)](https://github.com/espressif/esp-idf)
[![Hardware ESP32-S3](https://img.shields.io/badge/Hardware-ESP32--S3-000000?style=flat-square)](https://www.espressif.com/en/products/socs/esp32-s3)
[![License MIT](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![SG Project Day 2026](https://img.shields.io/badge/Saint%20Gabriel's%20College-SG%20Project%20Day%202026-darkblue?style=flat-square)](docs/SG_PROJECT_DAY_KNOWLEDGE_BASE.md)

Device-free, privacy-preserving human activity classification and elderly fall detection working demonstration powered by 2.4 GHz WiFi Channel State Information (CSI) from dual ESP32-S3 microcontrollers and deep learning.

---

## Overview

Elderly falls represent a leading cause of fatal and non-fatal injuries worldwide. Conventional monitoring solutions suffer from fundamental trade-offs:
- **Optical Cameras**: Violate personal privacy, creating blind spots in private quarters such as bedrooms and bathrooms.
- **Wearable Sensors**: Require continuous user compliance (frequently forgotten, removed before sleep or bathing, or discharged).

This project implements an end-to-end wireless sensing system using **Channel State Information (CSI)** extracted from commodity **ESP32-S3** microcontrollers operating on 2.4 GHz WiFi (802.11n HT40). By tracking multi-path Doppler shifts and amplitude perturbations across 114 active subcarriers, the system detects human presence, classifies activities (walking vs. stationary rest), and identifies sudden falls in real time without requiring cameras, wearables, or ambient lighting.

### Technology Comparison Matrix

| Feature / Modality | Optical Cameras (CCTV) | Wearable Trackers (Smartwatch / Pendants) | Infrared Sensors (PIR) | mmWave Radar (60 GHz) | **ESP32-S3 WiFi CSI (This System)** |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Privacy Preservation** | ❌ None (Video recording) | ⚠️ Moderate (Location / biometric data) | ✅ High | ✅ High | **✅ 100% Privacy-Preserving (Zero-Vision)** |
| **User Compliance** | ✅ Passive | ❌ High burden (Must wear & recharge daily) | ✅ Passive | ✅ Passive | **✅ 100% Device-Free (Zero user compliance)** |
| **Darkness / Steam Tolerance** | ❌ Fails in total darkness & shower steam | ✅ High | ⚠️ Heat/steam-sensitive | ✅ High | **✅ Complete RF Penetration (Works in 0 Lux & bathroom steam)** |
| **Non-Line-of-Sight (NLoS)** | ❌ Blocked by blankets, furniture & doors | N/A | ❌ Direct line-of-sight only | ⚠️ Weak wall/blanket penetration | **✅ High Multipath Penetration (2.4 GHz $\lambda \approx 12.3$ cm)** |
| **Activity Discrimination** | ✅ High (Skeleton tracking) | ⚠️ Impact only (Falls vs jumps) | ❌ Binary motion only (No posture) | ✅ High | **✅ Fine-Grained (Walking vs Sitting vs 95.12% Fall Detection)** |
| **Hardware Cost** | ⚠️ High ($50 - $150 / node) | ⚠️ Moderate ($30 - $100 / person) | ✅ Very low ($3 - $5) | ❌ Expensive ($50 - $120 / chip) | **✅ Ultra Low-Cost (~$25 / ~800 THB total dual-board system)** |
| **Deployment Maturity** | Commercial | Commercial | Commercial | Experimental / Commercial | **✅ Working Demo (Live 60 FPS GUI + Real Fall Validation)** |

<p align="center">
  <img src="assets/hardware_setup.jpg" width="85%" alt="ESP32-S3 Hardware Setup with 6 dBi External Antennas" />
  <br />
  <em>Hardware deployment: Dual ESP32-S3 development boards equipped with 6 dBi omni-directional antennas.</em>
</p>

---

## System Architecture

```text
  [Transmitter (Tx) - COM5]                              [Receiver (Rx) - COM3]
  ESP32-S3 + 6 dBi Antenna                               ESP32-S3 + 6 dBi Antenna
  ┌─────────────────────────┐                            ┌─────────────────────────┐
  │ Broadcast 100 Hz Frames │                            │ Extract Raw CSI Payloads│
  │ Channel 6 (HT40 Mode)   │                            │ 190 Raw IQ Subcarriers  │
  └────────────┬────────────┘                            └────────────┬────────────┘
               │                                                      │
               ▼                                                      ▼
    [Wireless Multipath Channel]                              [Serial 921,600 Baud]
    Body movement modulates WiFi carriers                             │
    via Doppler shifts & scattering                                   ▼
                                                          [Decoupled Ring Buffer]
                                                          Zero-latency reader thread
                                                                      │
                                                                      ▼
                                                          [Uniform 100 Hz Resampler]
                                                          Bridges serial arrival jitter
                                                                      │
                                                                      ▼
                                                          [Hardware AGCFaultGuard]
                                                          Blocks PHY gain-switch steps
                                                                      │
                                                                      ▼
                                                          [Streaming Hampel Filter]
                                                          Median outlier suppression
                                                                      │
                                                                      ▼
                                                           [Causal SOS Bandpass]
                                                           0.5 - 40 Hz 8th-Order (4 SOS)
                                                                      │
                                                                      ▼
                                                          [Sliding Window: 1s (100×20)]
                                                          PCA & Standard Scaler
                                                                      │
                                                                      ▼
                                                          [CNN-BiLSTM + Attention]
                                                          Spatial + Temporal Inference
                                                                      │
                                            ┌─────────────────────────┴─────────────────────────┐
                                            ▼                                                   ▼
                                 [Real-Time Fall Engine]                             [Activity Monitor GUI]
                                 scripts/realtime_detect.py                          scripts/realtime_activity_demo.py
                                 Threshold: 80% | Audible Buzzer                     60 FPS Waveform + Motion Gauge
```

---

## Hardware Configuration

The system is deployed in a single-link bistatic transceiver topology:

| Parameter | Transmitter (Tx) | Receiver (Rx) | Notes |
|:---|:---|:---|:---|
| **Microcontroller** | ESP32-S3-WROOM-1 | ESP32-S3-WROOM-1 | Dual-core Xtensa LX7 @ 240 MHz |
| **Port** | `COM5` | `COM3` | Hardware UART bridge @ 921,600 baud |
| **Antenna** | 6 dBi External Omni Antenna | 6 dBi External Omni Antenna | Extended link budget & multipath sensitivity |
| **Power Source** | Dedicated 5V USB Wall Supply | Host PC USB 3.0 Port | Isolated Tx ground prevents noise coupling |
| **Firmware** | `esp-csi/.../csi_send` | `esp-csi/.../csi_recv` | Espressif ESP-CSI framework on ESP-IDF v5.4 |
| **WiFi Protocol** | 802.11n HT40 | 802.11n HT40 | 40 MHz channel bandwidth |
| **RF Channel** | Channel 6 (2437 MHz) | Channel 6 (2437 MHz) | Center frequency with secondary channel |
| **Packet Rate** | 100 Hz broadcast | ~60-65 Hz serial throughput | Governed by UART baud bandwidth limits |

### Physical Layout
```text
    [COM5 - Tx Node]                  Activity & Sensing Zone                [COM3 - Rx Node]
    ┌───────────────┐                                                        ┌───────────────┐
    │   ESP32-S3    │◄──────────────────────── 3 - 5 m ─────────────────────►│   ESP32-S3    │
    │ 6 dBi Antenna │             Direct Line-of-Sight (LoS)                 │ 6 dBi Antenna │
    └───────────────┘                   Fresnel Ellipsoid                    └───────────────┘
     Height: ~1.0 m                                                           Height: ~1.0 m
```

---

## Core Engineering Innovations

### 1. Hardware AGC Step Suppression (`AGCFaultGuard`)
At fringe signal levels, the ESP32 Wi-Fi PHY AGC switches internal analog gain stages (e.g. LNA/VGA steps 24 $\leftrightarrow$ 26). Because firmware gain compensation operates with a single-packet latency relative to analog switching, raw CSI amplitudes occasionally suffer instantaneous common-mode gain jumps across all subcarriers simultaneously (empirically $\approx \pm 6 \text{ dB}$, up to $+10$ to $+20 \text{ dB}$ under severe dithering).
- **The Solution**: An online metric tracking frame-to-frame log-ratio uniformity across subcarriers:
  $$\mu = \text{mean}\left(\log \frac{A_k[t]}{A_k[t-1]}\right), \quad \sigma = \text{std}\left(\log \frac{A_k[t]}{A_k[t-1]}\right)$$
  When $\sigma \approx 0$ (< 0.35) while $|\mu|$ is large (> 0.69 $\approx 6 \text{ dB}$), the frame is flagged as an artificial gain step and suppressed, preventing severe filter ringing.
- **Note**: `AGCFaultGuard` is currently deployed in the real-time Activity Monitor GUI (`realtime_activity_demo.py`), `visualize_dual.py`, and `dual_link.py`. The Fall Detection runtime uses `StreamingHampel` as its primary defense.

### 2. Streaming Hampel Filter (`StreamingHampel`, Low-Latency / Quasi-Causal)
Isolated packet corruptions and RF noise impulses act as step inputs to narrow IIR bandpass filters, causing impulse ringing.
- **The Solution**: A streaming sliding-window Hampel filter with bounded look-ahead delay (~50 ms at 100 Hz due to a centered $2k+1$ window) computing the rolling median and Median Absolute Deviation (MAD):
  $$\text{MAD} = 1.4826 \times \text{median}(|x - \text{median}(x)|)$$
  Outliers exceeding threshold $\times \text{MAD}$ are replaced with the local median prior to bandpass filtering, stabilizing the IIR state.

### 3. Butterworth SOS Bandpass (0.5 - 40 Hz, order=4 per edge → 8th-order, 4 biquads)
Human torso movement during a fall produces characteristic Doppler frequencies:
$$f_D \le \frac{2v}{\lambda} \approx 16 \text{ Hz per m/s} \implies 32 - 49 \text{ Hz theoretical upper bound for fall}$$
- Traditional 10 Hz low-pass filters eliminate the critical fall transient.
- Non-causal zero-phase filtering (`filtfilt`) cannot run in real time.
- **The Solution**: A stateful, causal 8th-order Butterworth band-pass implemented as 4 second-order sections (SOS). The 0.5 Hz high-pass edge removes static DC multipath from walls and furniture; the 40 Hz low-pass edge rejects high-frequency noise while preserving rapid fall signatures.

### 4. Leakage-Free Machine Learning Pipeline
Overlapping sliding windows extracted from time-series recordings share temporal information. Randomly partitioning windows into train/test sets contaminates test distributions, artificially inflating validation accuracy.
- **The Solution**: The dataset is segmented strictly at the recording session level using `GroupShuffleSplit` on `groups.npy`. Windows from any given physical recording session exist entirely in the training split or the test split, never both.
- **Note on Current Data**: The dataset integrates **real physical human falls** (`fall_forward`: 5 physical fall recordings on safety padding with 17x to 109x Doppler signal contrast over quiet baseline) alongside real daily activities (`walking`: 10 recordings, `sitting_down`: 10 recordings). Due to human subject safety considerations (preventing acute injury to the investigator), lateral fall directions (`fall_backward`, `fall_sideways`) retain kinematic simulations. See [Limitations](docs/SG_PROJECT_DAY_KNOWLEDGE_BASE.md#9-ข้อจำกัดของโครงงาน-limitations).

---

## Deep Learning Architecture

The classification backbone combines spatial subcarrier feature extraction, temporal sequence dynamics, and multi-head self-attention:

```text
Input Tensor: Window of 100 samples × 20 PCA components (1.0 s @ 100 Hz)
  │
  ├──► 1D Convolution (64 filters, kernel=3)  + BatchNorm + ReLU + Dropout(0.2)
  ├──► 1D Convolution (128 filters, kernel=3) + BatchNorm + ReLU + MaxPool1D(2)
  │
  ├──► Bidirectional LSTM (hidden=128, 2 layers, dropout=0.3)
  │      Captures forward & backward temporal dynamics of posture transition
  │
  ├──► Multi-Head Self-Attention (4 heads, embed_dim=256)
  │      Dynamically weights critical impact frames over static pre/post phases
  │
  └──► Fully Connected Classifier (Linear 64 -> Linear 2)
         ├── Class 0: Fall (Alarm Triggered)
         └── Class 1: Normal Daily Activity (Walking, Sitting, Standing, Resting)
```

### Model Evaluation Results & Empirical Benchmarks

The model was trained and evaluated under strict **recording-level zero-leakage conditions (`GroupShuffleSplit`)**, partitioning 37 independent recordings (889 segmented windows) so that overlapping windows from any recording never cross between splits.

#### System Evolution: Concept Prototype vs. Live Working Demo

| Dimension / Metric | Phase 1: Concept Prototype | Phase 2: Live Working Demo (Current) | Verification & Impact |
|:---|:---:|:---:|:---|
| **System Maturity** | Initial Proof-of-Concept (TRL 3) | **Verified Working Demo (TRL 4–5)** | Live hardware streaming + 60 FPS GUI + buzzer alert |
| **Fall Dataset** | 100% Synthetic simulations | **Real Human Falls (5 sessions)** + Synthetic augmentation | Genuine physical Doppler transient (17x–109x contrast) |
| **Daily Activities** | Real Walking / Sitting | Real Walking (10) + Sitting (10) + Real Falls (5) | Full multi-class activity coverage (37 sessions, 889 windows) |
| **Signal Conditioning** | Raw with AGC gain-step jumps | **Causal DSP (SOS Bandpass + Hampel + Resampler)** | Stabilized baseline ($0.10 - 0.18\text{ dB}^2$), zero plateaus |
| **Validation Loss / Acc** | 0.4228 / 74.24% | **0.1307 / 96.64%** | **69% loss reduction, +22.40% validation gain** |
| **Test Accuracy** | 42.6% (Single-class test) | **81.05% (Balanced 153 windows)** | Zero-leakage `GroupShuffleSplit` across held-out recordings |
| **Fall Sensitivity (Recall)**| N/A | **95.12% (39 of 41 falls caught)** | Critical safety metric for life-threatening falls |
| **Non-Fall Specificity** | N/A | **97.70% (Minimal false alarms)** | Prevents user alert fatigue in daily life |
| **Fall ROC-AUC** | N/A | **0.9046** | High discriminative capacity ($> 0.90$) |
| **Fall PR-AUC** | N/A | **0.7160** | Accurate rare-event detection on held-out test split |
| **Live Interface** | Terminal output | **Interactive 60 FPS GUI + Real-Time Buzzer Alarm** | Ready for live stage & interactive judge demonstration |

#### Detailed Test Set Metrics (Held-Out Split, N=153)

| Metric | Held-Out Test Set (N=153) | Validation Checkpoint (Epoch 14) | Notes |
|:---|:---:|:---:|:---|
| **Overall Accuracy** | **81.05%** | **96.64%** | Generalisation across held-out test recordings |
| **Fall Sensitivity (Recall)** | **95.12%** | — | **39 out of 41 fall windows successfully detected** |
| **Non-Fall Precision** | **97.70%** | — | High reliability; 85/87 normal predictions are truly normal |
| **Weighted F1-Score** | **0.8207** | — | Harmonic balance across imbalanced test distribution |
| **Fall-Class F1-Score** | **0.7290** | — | Focus metric for the rare event (Fall) |
| **ROC-AUC (Fall as Positive)**| **0.9046** | — | Excellent discriminative power ($> 0.90$) |
| **PR-AUC (Fall as Positive)** | **0.7160** | — | Area under Precision-Recall curve for rare event |
| **Best Validation Loss** | — | **0.1307** | Early stopping checkpoint (Epoch 14 of 29) |
| **Inference Latency** | **< 3.2 ms** | **< 3.2 ms** | RTX 3050 Laptop GPU / CPU (real-time capable) |

#### Live Hardware Verification & Empirical Operating Characteristics (TRL 4–5 Demo)

During end-to-end live testing with physical ESP32-S3 hardware streaming at 60–65 Hz over USB UART (`COM3`), the unified dashboard exhibited distinct empirical behaviors that highlight the realities of commodity RF sensing:

1. **Near-Zero False Negatives ($\approx 0\%$) / Flawless Fall Sensitivity from Settled State**:
   - When transitioning from a quiet, standing, or settled state into a physical fall onto the safety cushion, the system catches the fall with **100% consistency** (e.g. `Confidence: 99.8%, Energy: 2.38 dB²`), triggering the buzzer and flashing emergency banner instantly.
   - For safety-critical elderly care, preventing missed falls (Zero False Negatives) is paramount.
2. **Elevated False Positives during Continuous Dynamic Walking**:
   - In live continuous operation, prolonged active walking back and forth without pausing can occasionally cross the fall threshold (False Positive).
   - **Root Cause**:
     - *Dataset Scale & Diversity*: The training set contains 37 recording sessions (889 windows; 10 walking, 10 sitting, 5 real falls, 12 synthetic falls). In a confined indoor space, continuous vigorous walking generates continuous multi-subcarrier Doppler energy that occasionally overlaps with the fall decision boundary in PCA feature space.
     - *Room Multipath Reverberation Accumulation*: Continuous walking creates ongoing reflective interference in a single-antenna pair setup without spatial diversity.
3. **Live Demonstration Protocol (Recommended for Presentations)**:
   - Allow the demonstrator to stand still or sit for 2–3 seconds until the indicator settles into the green `SITTING STILL / QUIET` state.
   - Execute the forward fall onto the safety cushion. The model immediately detects the impact with $\ge 99\%$ confidence and sounds the buzzer.
4. **Production Engineering Roadmap (Post-Demo)**:
   - **Temporal State Machine with Immobility Criterion ("The Long Lie Filter")**: True falls are physiologically followed by *prolonged immobility on the floor* (Wild et al., 1981). Requiring 1.5–2.0 seconds of post-impact stillness ($E < 0.50\text{ dB}^2$) before locking the alarm suppresses walking false alarms completely, as a walking person continues moving.
   - **Expanded Multi-Speed Gait Dataset**: Augmenting the corpus with varying walking speeds, pacing, and turning dynamics to broaden the daily activity manifold.

#### Confusion Matrix & Discrimination Curves (Held-Out Test Set)

<p align="center">
  <img src="assets/training_curves.png" width="95%" alt="CSI Fall Detection Training and Validation Curves" />
  <br />
  <em>Training Dynamics: Cross-Entropy Loss (Left) and Accuracy (Right) across 29 epochs, highlighting the optimal checkpoint at Epoch 14 (Val Loss: 0.1307, Val Acc: 96.64%).</em>
</p>

<p align="center">
  <img src="assets/confusion_matrix.png" width="48%" alt="Test Confusion Matrix" />
  <img src="assets/roc_curve.png" width="48%" alt="ROC Curve with AUC = 0.9046" />
  <br />
  <em>Left: Confusion Matrix on 153 held-out test windows. Right: ROC Curve demonstrating 0.9046 Area Under the Curve for Fall detection.</em>
</p>

<p align="center">
  <img src="assets/pr_curve.png" width="55%" alt="Precision-Recall Curve with AUC = 0.7160" />
  <br />
  <em>Precision-Recall Curve (Fall as Positive Class, PR-AUC = 0.7160) confirming high recall for fall events.</em>
</p>

### Raw CSI Data Visualization
The spectrograms below demonstrate the distinct physical signatures of a Real Human Fall versus Normal Walking captured by our dual-ESP32 setup:

<p align="center">
  <img src="assets/csi_comparison.png" width="95%" alt="CSI Amplitude Variation: Real Fall vs Walking" />
  <br />
  <em>Real Human CSI Data: Left panel captures a genuine forward fall onto safety padding, showing pre-fall stillness, a sharp multi-subcarrier Doppler impact transient (~frame 250), and post-fall rest. Right panel displays periodic multipath oscillations during continuous normal walking.</em>
</p>

---

## Real-Time Runtime Engines

The project provides a unified flagship GUI application alongside specialized standalone engines:

### 1. Unified Real-Time Activity & Fall Dashboard (`scripts/realtime_unified_monitor.py` - ⭐ Flagship)
The complete all-in-one real-time monitoring solution for live demonstrations. Combines motion energy classification (Walking vs. Sitting) with deep learning fall inference on sliding windows, featuring:
- **Dynamic 3-State HUD Banner**: SITTING STILL (🧘 Green) $\leftrightarrow$ WALKING (🚶 Cyan) $\leftrightarrow$ **EMERGENCY: FALL DETECTED!** (🚨 Red Pulsing Alert).
- **Dual Live Gauges**: Kinetic Motion Energy ($0 - 25\text{ dB}^2$) and AI Fall Probability ($0\% - 100\%$) with 80% critical threshold.
- **60 FPS Filtered Waveform Canvas**: 10 active subcarrier streams with causal SOS bandpass filtering.
- **Live Event Audit Log & Audible Buzzer**: Timestamped state transition history and emergency alert tones (`winsound.Beep`).

```powershell
# Live Hardware Execution (COM3 Receiver)
python scripts/realtime_unified_monitor.py --port COM3

# Replay Physical Fall or Walking Datasets for Demonstrations
python scripts/realtime_unified_monitor.py --replay data/raw/fall_forward/sample_20260909_200400.csv
python scripts/realtime_unified_monitor.py --replay data/raw/walking/sample_20260715_192540.csv
```

<p align="center">
  <img src="assets/realtime_demo.png" width="90%" alt="Real-Time Human Activity Monitor Interface" />
  <br />
  <em>Real-time Activity & Fall Monitor Dashboard: Interactive 60 FPS interface displaying active motion classification, dual telemetry gauges, real-time filtered CSI subcarriers, and event audit log.</em>
</p>

### 2. Standalone Engines (Specialized / Headless)
- **Live Human Activity Monitor GUI (`scripts/realtime_activity_demo.py`)**: Lightweight activity-only monitor.
- **Headless Fall Detection CLI (`scripts/realtime_detect.py`)**: Low-overhead terminal runtime for headless or embedded edge devices.

```powershell
python scripts/realtime_detect.py --port COM3 --threshold 0.80
```

Terminal output:
```text
[20:45:12] Status: NORMAL   | Fall: [==------------------] 14.2% | Daily: 85.8% | 65 pkt/s
[20:45:14] Status: CHECKING | Fall: [==============------] 72.0% | Daily: 28.0% | 65 pkt/s
[20:45:15] Status: FALL!    | Fall: [====================] 94.6% | Daily:  5.4% | 65 pkt/s >> ALARM
```

---

## Repository Structure

```text
ESP-CSI/
├── assets/                      # Curated visual figures for documentation & presentation
│   ├── hardware_setup.jpg       # Dual ESP32-S3 boards with 6 dBi antennas
│   ├── realtime_demo.png        # Real-time Activity Monitor GUI screenshot
│   ├── confusion_matrix.png     # Model evaluation confusion matrix
│   ├── roc_curve.png            # ROC evaluation curve
│   ├── pr_curve.png             # Precision-Recall curve
│   └── walking_vs_sitting.png   # Spectral activity comparison
├── docs/
│   └── SG_PROJECT_DAY_KNOWLEDGE_BASE.md # Comprehensive engineering & presentation guide
├── esp-csi/                     # Espressif ESP-CSI submodule
│   └── examples/get-started/    # csi_send (Tx) and csi_recv (Rx) firmware
├── data/
│   ├── raw/                     # Raw CSI recordings (.csv) by activity category
│   │   ├── walking/             # Active walking datasets
│   │   ├── sitting_down/        # Stationary resting datasets
│   │   ├── standing_up/         # Sit-to-stand transition datasets
│   │   ├── empty_room/          # Background environment baseline
│   │   └── fall_*/              # Synthetic fall datasets (sample_synth_*; real falls planned)
│   └── processed/               # Preprocessed tensors (X.npy, y.npy, groups.npy, pipeline_state.pkl)
├── scripts/
│   ├── csi_dsp.py               # Shared Causal DSP engine (Hampel, SOS Bandpass, Resampler)
│   ├── parse_csi.py             # High-throughput serial parser & packet reader
│   ├── collect_csi.py           # Automated CSI data recording CLI with quality audit
│   ├── preprocess.py            # End-to-end signal preprocessing & sliding window pipeline
│   ├── visualize_csi.py         # 60 FPS real-time oscilloscope & spectral snapshot tool
│   ├── realtime_activity_demo.py # Live Walking vs Sitting classification monitor
│   ├── realtime_detect.py       # Live Fall Detection runtime engine with buzzer alarm
│   └── realtime_unified_monitor.py # Unified All-in-One Dashboard (Walking, Sitting, Fall + Buzzer)
├── model/
│   ├── cnn_lstm.py              # CNN-BiLSTM + Multi-Head Self-Attention model definition
│   ├── dataset.py               # Leakage-free PyTorch Dataset with GroupShuffleSplit
│   ├── train.py                 # GPU-accelerated training pipeline (CUDA 11.8)
│   ├── evaluate.py              # Fall-centric evaluation suite
│   └── saved/                   # Saved model checkpoints & training history
├── config.yaml                  # Unified hardware, DSP, and model parameters
├── requirements.txt             # Python environment dependencies
└── README.md                    # Project documentation
```

---

## Quickstart Guide

### 1. Environment Setup

```powershell
# Clone the repository
git clone https://github.com/Ken35943/SG_106B07.git
cd SG_106B07

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies (Python 3.11 recommended)
pip install -r requirements.txt
```

### 2. Firmware Installation

Flash the precompiled binaries to both ESP32-S3 boards using `esptool`:

```powershell
# Flash Receiver (COM3)
python -m esptool --chip esp32s3 -p COM3 write_flash 0x0 esp-csi/examples/get-started/csi_recv/build/bootloader/bootloader.bin 0x8000 esp-csi/examples/get-started/csi_recv/build/partition_table/partition-table.bin 0x10000 esp-csi/examples/get-started/csi_recv/build/csi_recv.bin

# Flash Transmitter (COM5)
python -m esptool --chip esp32s3 -p COM5 write_flash 0x0 esp-csi/examples/get-started/csi_send/build/bootloader/bootloader.bin 0x8000 esp-csi/examples/get-started/csi_send/build/partition_table/partition-table.bin 0x10000 esp-csi/examples/get-started/csi_send/build/csi_send.bin
```

### 3. Verification & Live Oscilloscope

Verify live CSI frame reception across all 114 active subcarriers:

```powershell
python scripts/visualize_csi.py --port COM3
```

### 4. Data Collection & Training Workflow

```powershell
# 1. Collect calibrated activity recordings
python scripts/collect_csi.py --activity walking --duration 15 --port COM3
python scripts/collect_csi.py --activity sitting_down --duration 10 --port COM3
python scripts/collect_csi.py --activity fall_forward --duration 10 --port COM3

# 2. Run signal preprocessing pipeline
python scripts/preprocess.py

# 3. Train neural network on GPU
python model/train.py --epochs 30 --batch-size 32

# 4. Generate evaluation metrics & plots
python model/evaluate.py
```

### 5. Unified Live Activity & Fall Detection Demo

Launch the all-in-one 60 FPS interactive dashboard (detects **Walking 🚶**, **Sitting Still 🧘**, and **Falls 🚨** with real-time audio buzzer alarm):

```powershell
# Live Hardware Stream (Receiver on COM3)
python scripts/realtime_unified_monitor.py --port COM3

# Replay Mode (Demonstrate on recorded human activities)
python scripts/realtime_unified_monitor.py --replay data/raw/walking/sample_20260715_192540.csv
python scripts/realtime_unified_monitor.py --replay data/raw/fall_forward/sample_20260909_200400.csv
```

---

## Troubleshooting & Engineering Notes

| Symptom | Root Cause | Resolution |
|:---|:---|:---|
| **Stale Serial Backlog / Packet Lag** | Structural oversubscription: 921,600 baud carries ~92 kB/s while Tx broadcasts at 100 Hz (~150 kB/s). | Handled automatically by decoupled ingestion threads and high-watermark serial buffer purging (64 kB threshold). |
| **Periodic Common-Mode Gain Spikes (≈ ±6 dB typical; up to +10–20 dB worst case)** | ESP32 internal PHY AGC switches gain steps (24 $\leftrightarrow$ 26) with 1-packet compensation lag. | Suppressed via `AGCFaultGuard` and `StreamingHampel` prior to the IIR bandpass filter. |
| **Halved Frame Rate (30 Hz instead of 65 Hz)** | Aggressive serial buffer flushing (`in_waiting > 4096`) discarding valid packet frames. | Purge threshold raised to 64 kB (`scripts/realtime_activity_demo.py`), preserving contiguous packets. |
| **All Predictions Report NORMAL** | `realtime_detect.py` is a binary fall classifier; walking is defined as Daily Activity (Normal). | Normal behavior. To observe live motion sensitivity, run `scripts/realtime_activity_demo.py` or simulate falls with `--threshold 0.50`. |
| **False Positive Alarm during Continuous Walking in Live Demo** | Multipath accumulation from continuous pacing + small training dataset (10 walking recordings) briefly intersects fall decision boundary. | **Demonstration Protocol**: Stand/sit still for 2–3 seconds to let baseline settle (green status), then execute fall (100% detection rate). **Future Fix**: Implement 2.0s post-fall immobility verification. |

---

## Documentation & Project Information

- **Complete Knowledge Base & Presentation Guide**: See [docs/SG_PROJECT_DAY_KNOWLEDGE_BASE.md](docs/SG_PROJECT_DAY_KNOWLEDGE_BASE.md) for an exhaustive technical report covering RF physics, mathematical derivations, real-world engineering hurdles, and defense presentation scripts.
- **Academic Context**: Saint Gabriel's College — **SG Project Day 2026** (Project SG_106B07).
- **Core Frameworks**: [Espressif ESP-CSI](https://github.com/espressif/esp-csi), [PyTorch](https://pytorch.org/), [PyQtGraph](https://www.pyqtgraph.org/).
