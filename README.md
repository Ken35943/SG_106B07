# SG_106B07 — ESP32-S3 WiFi CSI Fall Detection System

High-precision, privacy-preserving elderly fall detection using WiFi Channel State Information (CSI) from ESP32-S3 boards and deep learning accelerated with NVIDIA RTX GPUs.

---

## 🌟 Highlights & Architecture Upgrades

- **Real-Time 60 FPS Oscilloscope & Snapshot UI**: Decoupled serial streaming thread, zero-latency pre-allocated ring buffer, and fast neon stem rendering (`scripts/visualize_csi.py`).
- **Real-Time Activity Monitor (`scripts/realtime_activity_demo.py`)**: Live visual classification between *Walking* and *Sitting Still* with a dynamic motion energy gauge.
- **Shared Causal DSP Core (`scripts/csi_dsp.py`)**: 
  - Vectorized Hampel outlier filter (**196× speedup** over iterative Python).
  - 114-occupied subcarrier mapping for HT40 (dropping null/guard bands).
  - Causal 4th-order SOS Bandpass filter (0.5–40 Hz) preserving 15–30 Hz fall Doppler signatures.
  - Streaming uniform resampler handling real-world serial jitter.
- **Leakage-Free Dataset Pipeline**: Uses `GroupShuffleSplit` on recording groups (`groups.npy`) ensuring overlapping windows never contaminate test splits.
- **NVIDIA CUDA Acceleration**: Fully supports PyTorch 2.7+ with CUDA 11.8 on NVIDIA GeForce RTX 3050 Laptop GPUs (5 epochs in ~1.9s).
- **Fall-Centric Evaluation**: Explicitly treats Fall (Class 0) as the positive class for PR-AUC, ROC-AUC, and F1-score.

---

## 📡 Hardware Setup

| Board | Port | Antenna | Role | Firmware |
|:---|:---|:---|:---|:---|
| ESP32-S3 #1 | **COM3** | 6 dBi External Antenna | **CSI Receiver (Rx)** | `esp-csi/examples/get-started/csi_recv` |
| ESP32-S3 #2 | **COM5** | Onboard PCB Antenna | **CSI Sender (Tx)** | `esp-csi/examples/get-started/csi_send` |

### Room Layout
```text
    [COM5 - Sender]                   Activity Zone                  [COM3 - Receiver]
    ┌─────────────┐                                                  ┌─────────────┐
    │  ESP32-S3   │◄─────────────────── 3–5 m ──────────────────────►│  ESP32-S3   │
    │ PCB Antenna │           Person moves or rests here             │ 6dBi Antenna│
    └─────────────┘                                                  └─────────────┘
    Height: ~1.0 m                                                   Height: ~1.0 m
```

---

## 📁 Repository Structure

```text
ESP-CSI/
├── esp-csi/                     # Espressif ESP-CSI framework submodule
├── data/
│   ├── raw/                     # Raw CSI session recordings (.csv)
│   │   ├── walking/             # Walking / active motion samples
│   │   ├── sitting_down/        # Sitting / stationary samples
│   │   ├── standing_up/         # Sit-to-stand transitions
│   │   ├── empty_room/          # Empty background room calibration
│   │   └── fall_*/              # Cushioned fall recordings
│   └── processed/               # Preprocessed tensors (X.npy, y.npy, groups.npy, pipeline_state.pkl)
├── scripts/
│   ├── csi_dsp.py               # Shared Causal DSP engine (Hampel, SOS Bandpass, STFT Doppler)
│   ├── parse_csi.py             # Serial protocol packet parser & streaming reader
│   ├── collect_csi.py           # Interactive data collection CLI with quality grading
│   ├── preprocess.py            # End-to-end signal preprocessing & sliding window pipeline
│   ├── visualize_csi.py         # 60 FPS real-time CSI wave & spectrum visualizer
│   ├── realtime_activity_demo.py # Live Walking vs Sitting GUI with energy gauge
│   ├── realtime_detect.py       # Live Fall Detection runtime engine with audio alarm
│   └── visualize_heatmap.py     # Multi-node spatial energy heatmap
├── model/
│   ├── cnn_lstm.py              # CNN-BiLSTM + Multi-Head Self-Attention network
│   ├── dataset.py               # Leakage-free PyTorch Dataset with GroupShuffleSplit
│   ├── train.py                 # GPU-accelerated training pipeline (CUDA 11.8)
│   ├── evaluate.py              # Fall-centric evaluation (ROC-AUC, PR-AUC, Confusion Matrix)
│   └── saved/                   # Checkpoints & evaluation plots
├── config.yaml                  # Unified hardware, DSP, and model configuration
└── requirements.txt             # Python dependencies
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```powershell
# Recommended: Python 3.11 with PyTorch CUDA 11.8 on Windows
cd C:\Users\Ken\Documents\ESP-CSI
pip install -r requirements.txt
```

### 2. Verify CSI Connection & View Live Waves
Connect both ESP32-S3 boards (COM3 = Receiver, COM5 = Sender):
```powershell
# Live 60 FPS oscilloscope:
python scripts/visualize_csi.py --port COM3

# Or replay an existing capture without hardware:
python scripts/visualize_csi.py --replay data/raw/walking/sample_20260715_192540.csv
```

### 3. Test Live Activity Classification (Walking vs. Sitting)
Test real-time human motion detection in your room:
```powershell
# Live interactive activity monitor:
python scripts/realtime_activity_demo.py --port COM3

# Or replay recorded sessions:
python scripts/realtime_activity_demo.py --replay data/raw/walking/sample_20260906_205635.csv
python scripts/realtime_activity_demo.py --replay data/raw/sitting_down/sample_20260906_210020.csv
```

---

## 🏃 Data Collection & Model Training Workflow

### Step 1: Collect Real Activities
Use a soft mattress/cushion for fall exercises.

```powershell
# Normal daily activities:
python scripts/collect_csi.py --activity walking --duration 15 --port COM3
python scripts/collect_csi.py --activity sitting_down --duration 10 --port COM3
python scripts/collect_csi.py --activity standing_up --duration 10 --port COM3
python scripts/collect_csi.py --activity empty_room --duration 10 --port COM3

# Cushioned falls:
python scripts/collect_csi.py --activity fall_forward --duration 10 --port COM3
python scripts/collect_csi.py --activity fall_backward --duration 10 --port COM3
```

### Step 2: Preprocess Data
Applies uniform resampling, log-amplitude conversion, vectorized Hampel filtering, and causal bandpass (0.5–40 Hz):
```powershell
python scripts/preprocess.py
```

### Step 3: Train on GPU (NVIDIA RTX 3050)
```powershell
python model/train.py --epochs 30 --batch-size 32
```

### Step 4: Evaluate Model Performance
Inspect generated confusion matrices and ROC/PR curves saved in `model/saved/evaluation/`:
```powershell
python model/evaluate.py
```

### Step 5: Run Real-Time Fall Detection Engine
```powershell
# Live detection with audible buzzer alarm:
python scripts/realtime_detect.py --port COM3

# Offline test on recorded file:
python scripts/realtime_detect.py --replay data/raw/walking/sample_20260715_192540.csv

# Simulated streaming mock:
python scripts/realtime_detect.py --mock
```

---

## 🧠 Model Architecture

The deep learning architecture combines spatial feature extraction, temporal sequence modelling, and self-attention:

```text
Input Window (Batch, 100, 20)
  │
  ├──► Conv1D(filters=64, kernel=3)  + BatchNorm + ReLU + Dropout
  ├──► Conv1D(filters=128, kernel=3) + BatchNorm + ReLU + MaxPool1D(2)
  │
  ├──► Bidirectional LSTM (hidden=128, 2 layers, dropout=0.3)
  │
  ├──► Multi-Head Self-Attention (4 heads, embeds multi-path phase changes)
  │
  └──► Fully Connected Classifier (Linear 64 -> Linear 2 classes)
         ├── Class 0: Fall (Alarm Triggered)
         └── Class 1: Normal Daily Activity (Walking, Sitting, Standing, Idle)
```

---

## 📚 References & Acknowledgments

- **Espressif Systems**: [ESP-CSI Framework](https://github.com/espressif/esp-csi) & ESP-IDF v5.4.
- **Signal Processing**: IEEE Transactions on Mobile Computing (WiFall, RT-Fall, FallDeFi).
- **Hardware Acceleration**: PyTorch CUDA on NVIDIA RTX Ampere Architecture.
