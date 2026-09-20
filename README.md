# ⚡ Ultra-Low Latency Invincible Rock-Paper-Scissors System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![Platform](https://img.shields.io/badge/Platform-Laptop%20%7C%20Raspberry%20Pi%205%20(ARM64)-orange.svg)]()
[![Inference Latency](https://img.shields.io/badge/Inference_Latency-%3C0.60ms_(CPU)-brightgreen.svg)]()
[![Validation Accuracy](https://img.shields.io/badge/Val_Accuracy-93.92%25_(Real_Webcam)-success.svg)]()

An ultra-low latency, computer-vision-driven **Invincible Rock-Paper-Scissors** system. 

Originally developed and validated on a standard laptop RGB webcam, the codebase is fully optimized for edge deployment on a **Raspberry Pi 5 (ARM64, Debian 12 Bookworm)** using a standard **USB webcam via `cv2.VideoCapture` (V4L2)** — no specialized CSI camera module or `picamera2` required.

Adapted from the neuromorphic Event Camera (DVS) project [SensorsINI/dextra-roshambo-python](https://github.com/SensorsINI/dextra-roshambo-python), this project replaces specialized neuromorphic event camera hardware with an optimized **Pseudo-Event Motion Masking Pipeline** (Frame Differencing + Calibrated Binary Thresholding), paired with a pure PyTorch **RoshamboNet v2 Tiny CNN** (with BatchNorm), an optional **ONNX Runtime** acceleration path for ARM Cortex-A76, and a **5-frame temporal majority filter**.

---

## 🎯 Key Features

- **Standard USB Webcam Input**: Relies entirely on standard USB UVC webcams via OpenCV's `cv2.VideoCapture` on both laptop and Raspberry Pi 5. No CSI ribbon cables or `picamera2` dependencies.
- **Pseudo-Event Motion Pipeline**: Replaces neuromorphic DVS event streams with consecutive frame differencing ($|I_t - I_{t-1}|$) and calibrated binary thresholding (`threshold=18`) to isolate sharp, noise-free motion trails in real time.
- **Dual Inference Engine (PyTorch & ONNX Runtime)**:
  - **PyTorch Eager CPU**: ~0.60 ms on standard laptop CPU.
  - **ONNX Runtime (ARM64 NEON)**: Highly optimized fallback for the Raspberry Pi 5's Cortex-A76 cores.
- **Pi 5 Multi-Core Thread Tuning**: Configurable thread allocation (`--threads 4`) prevents thread over-subscription and context-switching overhead on the Pi 5's 4 cores.
- **Robust V4L2 Camera Backend**: Auto-selects `cv2.CAP_V4L2` on Linux/Pi, forces a 1-frame driver buffer (`CAP_PROP_BUFFERSIZE=1`) to eliminate frame lag, and provides actionable diagnostics if a USB camera node is busy or disconnected.
- **Real-Time Latency Monitoring**: Automatically logs warnings if total per-frame processing exceeds a safe latency budget (`--latency_budget_ms 16.0`).
- **Motion Activity Curation**: Automated filtering (`min_gesture_pixels=15`) cleans zero-motion frames from gesture classes during bursts, ensuring clean label boundaries.
- **Invincible Counter Logic**: Immediately predicts user gesture onset and outputs the winning counter-move:
  - ✊ **Rock** $\rightarrow$ AI plays ✋ **Paper** *(Paper covers Rock)*
  - ✋ **Paper** $\rightarrow$ AI plays ✌️ **Scissors** *(Scissors cuts Paper)*
  - ✌️ **Scissors** $\rightarrow$ AI plays ✊ **Rock** *(Rock crushes Scissors)*
  - ⏳ **Background** $\rightarrow$ AI stays on **Standby**
- **5-Frame Temporal Majority Filter**: Ported from the original research repository to prevent transient flicker and ensure rock-solid prediction stability.

---

## 🏗️ Architecture Pipeline

```
  Standard USB Webcam (Laptop / Raspberry Pi 5 V4L2)
                           │
                           ▼
          Grayscale Conversion + Gaussian Blur
                           │
                           ▼
  Temporal Differencing: Diff = |Frame_t - Frame_t-1|
                           │
                           ▼
  Binary Thresholding (cv2.threshold @ 18 -> Motion Trail)
                           │
                           ▼
  Center ROI Crop & Resize to 64x64 Tensor [1, 1, 64, 64]
                           │
                           ▼
   RoshamboNet v2 (<1ms CPU Inference via PyTorch or ONNX)
                           │
                           ▼
  5-Frame Temporal Majority Filter (MajorityVote)
                           │
                           ▼
           Invincible Counter Decision Engine
```

---

## 📂 Project Structure

```
roshambo/
├── model.py                 # RoshamboNet v2 (BatchNorm) & MajorityVote filter
├── pseudo_event_producer.py # Robust V4L2 motion differencing & 64x64 mask generation
├── collect_data.py          # Interactive dataset generator with burst & motion filtering
├── train.py                 # Training pipeline (AdamW, CosineAnnealing, Augmentations)
├── export_onnx.py           # PyTorch -> ONNX export utility for Raspberry Pi 5
├── live_predict.py          # Real-time inference engine (PyTorch / ONNX Runtime + HUD)
├── motion_model.pth         # Trained PyTorch model weights (93.92% validation accuracy)
├── motion_model.onnx        # Exported ONNX model for high-efficiency Pi 5 inference
├── training_metrics.png     # Loss and accuracy progression curve
├── requirements.txt         # Python dependencies (aarch64 & x86_64 compatible)
└── dataset/                 # 64x64 binary motion event masks
    ├── 0_rock/
    ├── 1_paper/
    ├── 2_scissors/
    └── 3_background/
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/akibmehedi666/RPS-Semi-Arm.git
cd RPS-Semi-Arm
```

### 2. Dependencies & Environment

#### A. Raspberry Pi 5 (ARM64, Debian 12 Bookworm)

##### 1. Install System Dependencies via `apt`
Debian Bookworm requires system-level libraries for OpenCV and Video4Linux2:
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv v4l-utils libgl1 libglib2.0-0 libgomp1
```

> **Note on Debian Bookworm (PEP 668)**: Debian 12 Bookworm marks system Python as externally managed. Always install project packages inside a virtual environment.

##### 2. Create Virtual Environment & Install Python Packages
```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Package Availability on ARM64 (`aarch64`)**:
> - `torch` and `torchvision`: Official prebuilt `manylinux2014_aarch64` wheels on PyPI install directly with pip. No compilation from source required.
> - `numpy`: Official aarch64 wheels install automatically.
> - `onnxruntime`: Prebuilt aarch64 wheels on PyPI include optimized ARM NEON SIMD kernels for the Pi 5's Cortex-A76 cores.
> - `opencv-python`: Prebuilt wheels on PyPI work out of the box with `libgl1` and `libglib2.0-0`. If running purely headless over SSH without X11/Wayland display, you can alternatively install `opencv-python-headless`.

##### 3. USB Webcam Verification on the Pi
Check that your USB webcam is recognized by the V4L2 kernel driver:
```bash
v4l2-ctl --list-devices
```
Typically, a USB webcam registers as `/dev/video0` (stream) and `/dev/video1` (metadata), or `/dev/video2`/`/dev/video3`.

---

#### B. Standard Laptop (Linux / macOS / Windows)

**Windows (PowerShell / CMD):**
```powershell
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## 🎮 How to Run

### Step 1: Export Model to ONNX (Recommended for Pi 5)
Export the trained PyTorch checkpoint to an ONNX graph:
```bash
python3 export_onnx.py --model motion_model.pth --output motion_model.onnx
```

---

### Step 2: Run Real-Time Prediction

#### On Raspberry Pi 5 (with ONNX Runtime & 4-Core Tuning):
```bash
python3 live_predict.py --use_onnx --onnx_model motion_model.onnx --camera 0 --fps 30 --threads 4
```

#### On Laptop (Native PyTorch):
```bash
python3 live_predict.py --model motion_model.pth --camera 0 --fps 30 --threshold 18
```

- **Controls**:
  - Press **`r`** to reset the win streak counter.
  - Press **`q`** or **`ESC`** to exit and print performance metrics.
- **Headless / Benchmark Mode** (ideal for testing over SSH on the Pi):
  ```bash
  python3 live_predict.py --use_onnx --mock --benchmark 120
  ```

---

### Step 3: Collect Custom Gestures (Optional)
Position your hand inside the green square ROI and make gesture movements while pressing the burst keys:

**Windows:**
```powershell
python collect_data.py --camera 0 --threshold 18 --burst_size 25
```

**Linux / macOS:**
```bash
python3 collect_data.py --camera 0 --threshold 18 --burst_size 25
```

- Press **`r`** $\rightarrow$ Record 25-frame burst for **Rock**
- Press **`p`** $\rightarrow$ Record 25-frame burst for **Paper**
- Press **`s`** $\rightarrow$ Record 25-frame burst for **Scissors**
- Press **`b`** $\rightarrow$ Record 25-frame burst for **Background**
- Press **`q`** or **`ESC`** $\rightarrow$ Exit collection

*(Note: `--min_motion_pixels 15` automatically ignores blank frames before motion begins.)*

---

### Step 4: Clean & Sanitize Dataset (Optional but Recommended)
Filter out label leakage (e.g. closed fists in `1_paper`), empty frames, and duplicate sequential frames into `./dataset_quarantine`:

**Windows:**
```powershell
python clean_dataset.py
```

**Linux / macOS:**
```bash
python3 clean_dataset.py
```

- **Simulation Mode (Dry Run)**: Test thresholds without moving files:
  ```bash
  python3 clean_dataset.py --dry_run
  ```
- **Restore**: Undo quarantine and restore all files:
  ```bash
  python3 clean_dataset.py --restore
  ```

---

### Step 5: Train the RoshamboNet Model (Optional / On Laptop)
Training is typically performed on a laptop/desktop machine before deploying `motion_model.pth` or `motion_model.onnx` to the Pi:

**Windows:**
```powershell
python train.py --epochs 30 --batch_size 32 --lr 1e-3 --dataset_dir dataset --output_model motion_model.pth
```

**Linux / macOS:**
```bash
python3 train.py --epochs 30 --batch_size 32 --lr 1e-3 --dataset_dir dataset --output_model motion_model.pth
```

- **Optimizations**:
  - `AdamW(lr=1e-3, weight_decay=1e-4)` + `CosineAnnealingLR(T_max=30)`
  - Spatial augmentations: `RandomRotation(15)` + `RandomAffine(translate=(0.08, 0.08), scale=(0.95, 1.05))`
  - Class-weighted loss (`scissors: 1.2` to reward fine finger detection)
  - Saves the best checkpoint by validation accuracy to `motion_model.pth`
  - Generates training loss/accuracy curve in `training_metrics.png`


---

## 📊 Benchmark & Performance Results

### Inference Latency (Standard Laptop CPU & Raspberry Pi 5)
| Platform | Engine | Latency (Mean) | Latency (p95) | Sub-3ms Budget |
| :--- | :--- | :--- | :--- | :--- |
| **Laptop CPU** | PyTorch Eager | **0.60 ms** | **0.84 ms** | **PASSED** |
| **Laptop CPU** | ONNX Runtime | **0.42 ms** | **0.65 ms** | **PASSED** |
| **Raspberry Pi 5** | ONNX Runtime (4T) | **~1.20 ms** | **~1.80 ms** | **PASSED** |
| **Raspberry Pi 5** | PyTorch CPU (4T) | **~2.10 ms** | **~2.70 ms** | **PASSED** |

### Real Webcam Validation Accuracy (3,459 Curated Samples)
- **Overall Validation Accuracy**: **93.92%**

```
              precision    recall  f1-score   support

      0_rock       0.91      0.92      0.91       190
     1_paper       0.96      0.93      0.94       174
  2_scissors       0.93      0.94      0.93       172
3_background       0.96      0.99      0.97       155

    accuracy                           0.94       691
   macro avg       0.94      0.94      0.94       691
weighted avg       0.94      0.94      0.94       691
```

---

## 🛠️ Raspberry Pi 5 Camera Troubleshooting

If your USB webcam fails to open on the Pi:
1. **Check Video Nodes**:
   ```bash
   ls -l /dev/video*
   ```
2. **Identify Primary Video Stream Node**:
   ```bash
   v4l2-ctl --list-devices
   ```
   If `/dev/video0` is metadata, try `--camera 2` or `--camera 1`.
3. **Verify User Permissions**:
   ```bash
   sudo usermod -a -G video $USER
   ```
   (Log out and log back in for changes to take effect).
4. **Buffer & Latency Warning**:
   If terminal logs show `[PERF WARNING] Frame latency exceeds budget`, reduce resolution (`--width 320 --height 240`) or ensure `--use_onnx` is enabled.

---

## 📜 Acknowledgements & References

- Based on the neuromorphic event-camera research by **Sensors Group, INI (Institute of Neuroinformatics, UZH-ETH Zurich)**: [SensorsINI/dextra-roshambo-python](https://github.com/SensorsINI/dextra-roshambo-python).
- Original Authors: Tobi Delbruck et al.
