# ESP32-S3 CSI-Based Fall Detection System

AI-powered elderly fall detection using WiFi Channel State Information (CSI) from ESP32-S3.

## Overview

This project uses two ESP32-S3 development boards to capture WiFi CSI data and a CNN-LSTM-Attention deep learning model to detect falls in real-time. CSI captures how WiFi signals are affected by human movement — falls produce a distinctive "spike → stillness" signature that the AI model learns to recognize.

## Hardware Setup

| Board | COM Port | Antenna | Role |
|:------|:---------|:--------|:-----|
| ESP32-S3 #1 | **COM3** | 6dBi External | **CSI Receiver** (better antenna for signal quality) |
| ESP32-S3 #2 | **COM5** | PCB (onboard) | **CSI Sender** |

### Room Layout
```
    [COM5 - Sender]          Activity Zone           [COM3 - Receiver]
    ┌─────────┐                                      ┌─────────┐
    │ ESP32-S3│◄──────────── 3-5m ──────────────────►│ ESP32-S3│
    │ PCB ant.│        Person performs activities     │ 6dBi ant│
    └─────────┘          in this zone                └─────────┘
    ~1m height                                        ~1m height
```

## Project Structure

```
ESP-CSI/
├── esp-csi/                    # Espressif ESP-CSI framework (cloned)
├── firmware/
│   ├── csi_sender/             # Sender firmware (→ flash to COM5)
│   └── csi_receiver/           # Receiver firmware (→ flash to COM3)
├── data/
│   ├── raw/                    # Raw CSI recordings by activity type
│   └── processed/              # Preprocessed tensors (X.npy, y.npy)
├── scripts/
│   ├── parse_csi.py            # CSI data parsing utilities
│   ├── collect_csi.py          # Interactive data collection tool
│   ├── preprocess.py           # Signal preprocessing pipeline
│   └── visualize_csi.py        # Real-time CSI visualization
├── model/
│   ├── dataset.py              # PyTorch Dataset for CSI data
│   ├── cnn_lstm.py             # CNN-LSTM-Attention model architecture
│   ├── train.py                # Model training script
│   └── evaluate.py             # Evaluation and metrics
├── config.yaml                 # Central configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Quick Start

### 1. Prerequisites
- ESP-IDF v5.4+ installed ([Windows Installer](https://dl.espressif.com/dl/esp-idf/))
- Python 3.10+
- Two ESP32-S3 boards connected via USB

### 2. Install Python Dependencies
```bash
cd ESP-CSI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Flash Firmware
```bash
# Flash sender to COM5
cd esp-csi/examples/get-started/csi_send
idf.py set-target esp32s3
idf.py flash -b 921600 -p COM5

# Flash receiver to COM3
cd ../csi_recv
idf.py set-target esp32s3
idf.py flash -b 921600 -p COM3
```

### 4. Verify CSI Data
```bash
python scripts/visualize_csi.py --port COM3
```

### 5. Collect Training Data
```bash
python scripts/collect_csi.py --port COM3
```

### 6. Train Model
```bash
python scripts/preprocess.py
python model/train.py
```

### 7. Evaluate
```bash
python model/evaluate.py
```

## Model Architecture

**CNN-LSTM-Attention** — State-of-the-art for CSI-based fall detection (94-98% accuracy):

```
Input (batch, 100, 20) → Conv1D(64) → Conv1D(128) → MaxPool
    → BiLSTM(128, 2 layers) → Multi-Head Attention(4 heads)
    → FC(64) → Output(2 classes)
```

## References

- [Espressif ESP-CSI](https://github.com/espressif/esp-csi)
- [ESP-IDF Documentation](https://docs.espressif.com/projects/esp-idf/en/latest/)
- WiFall, RT-Fall, FallDeFi — Seminal CSI fall detection papers
