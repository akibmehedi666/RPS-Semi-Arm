# ⚡ Ultra-Low Latency Invincible Rock-Paper-Scissors System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![Inference Latency](https://img.shields.io/badge/Inference_Latency-%3C1.1ms_(CPU)-brightgreen.svg)]()
[![Validation Accuracy](https://img.shields.io/badge/Val_Accuracy-98.75%25-success.svg)]()

An ultra-low latency, computer-vision-driven **Invincible Rock-Paper-Scissors** system designed to run on a standard laptop RGB webcam. 

Adapted from the neuromorphic Event Camera (DVS) project [SensorsINI/dextra-roshambo-python](https://github.com/SensorsINI/dextra-roshambo-python), this project replaces specialized event camera hardware with an optimized **Pseudo-Event Motion Masking Pipeline** (Frame Differencing + Binary Motion Thresholding), paired with a pure PyTorch **RoshamboNet Tiny CNN** and a **5-frame temporal majority filter**.

---

## 🎯 Key Features

- **Pseudo-Event Motion Pipeline**: Replaces neuromorphic DVS event streams with consecutive frame differencing ($|I_t - I_{t-1}|$) and binary thresholding to isolate sharp, noise-free motion trails in real time.
- **Sub-3ms CPU Inference**: Powered by `RoshamboNet`, a lightweight 64×64 Tiny CNN ($114,436$ parameters) that completes inference in **~1.10 ms on standard CPU** without requiring a GPU.
- **Invincible Counter Logic**: Immediately predicts user gesture onset and outputs the winning counter-move:
  - ✊ **Rock** $\rightarrow$ AI plays ✋ **Paper** *(Paper covers Rock)*
  - ✋ **Paper** $\rightarrow$ AI plays ✌️ **Scissors** *(Scissors cuts Paper)*
  - ✌️ **Scissors** $\rightarrow$ AI plays ✊ **Rock** *(Rock crushes Scissors)*
  - ⏳ **Background** $\rightarrow$ AI stays on **Standby**
- **5-Frame Temporal Majority Filter**: Ported from the original research repository to prevent transient flicker and ensure rock-solid prediction stability.
- **Dual Display HUD**: Live webcam feed showing prediction confidence, AI counter announcement, streak tracking, and a scaled picture-in-picture preview of the 64×64 motion mask.

---

## 🏗️ Architecture Pipeline

```
  Standard Laptop Webcam (RGB 30–60 FPS)
                   │
                   ▼
  Grayscale Conversion + Gaussian Smoothing
                   │
                   ▼
  Temporal Differencing: Diff = |Frame_t - Frame_t-1|
                   │
                   ▼
  Binary Thresholding (cv2.threshold -> Sharp Motion Trail)
                   │
                   ▼
  Center ROI Crop & Resize to 64x64 Grayscale Tensor [1, 1, 64, 64]
                   │
                   ▼
  PyTorch RoshamboNet Tiny CNN (1.10 ms CPU Inference)
                   │
                   ▼
  5-Frame Temporal Majority Filter (voter)
                   │
                   ▼
  Invincible Counter Decision Engine
```

---

## 📂 Project Structure

```
roshambo/
├── model.py                 # RoshamboNet PyTorch architecture & MajorityVote filter
├── pseudo_event_producer.py # Webcam motion differencing & 64x64 mask generation
├── collect_data.py          # Interactive dataset generator with keyboard triggers ('r','p','s','b')
├── train.py                 # CPU-optimized training script (10-15 epochs in <15s)
├── live_predict.py          # Real-time inference engine with HUD visualizer
├── motion_model.pth         # Pre-trained model weights (98.75% validation accuracy)
├── training_metrics.png     # Loss and accuracy progression curve
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignored virtual environments, caches, and temp files
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
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🎮 How to Run

### Step 1: Collect Custom Gestures (Optional)
Position your hand inside the green square ROI and make gesture movements while pressing the burst keys:
```bash
python3 collect_data.py --camera 0
```
- Press **`r`** $\rightarrow$ Record 25-frame burst for **Rock**
- Press **`p`** $\rightarrow$ Record 25-frame burst for **Paper**
- Press **`s`** $\rightarrow$ Record 25-frame burst for **Scissors**
- Press **`b`** $\rightarrow$ Record 25-frame burst for **Background**
- Press **`q`** or **`ESC`** $\rightarrow$ Exit collection

*(To instantly generate a synthetic baseline dataset without camera input, run: `python3 collect_data.py --synthetic --synthetic_count 200`)*

---

### Step 2: Train the RoshamboNet Model
Train the 64×64 Tiny CNN on your dataset:
```bash
python3 train.py --epochs 12 --batch_size 32 --dataset_dir dataset --output_model motion_model.pth
```
- Trains for 12 epochs in **~14 seconds** on standard CPU.
- Evaluates classification metrics, prints a confusion matrix, and saves `training_metrics.png`.

---

### Step 3: Run the Real-Time Invincible System
Launch the real-time webcam inference system:
```bash
python3 live_predict.py --camera 0 --fps 60 --model motion_model.pth
```
- **Controls**:
  - Press **`r`** to reset the win streak counter.
  - Press **`q`** or **`ESC`** to exit and display latency performance statistics.
- **Headless / Benchmark Mode**:
  ```bash
  python3 live_predict.py --mock --benchmark 120
  ```

---

## 📊 Benchmark & Performance Results

### CPU Inference Latency (Intel / AMD CPU)
| Metric | Result | Target | Status |
| :--- | :--- | :--- | :--- |
| **Mean Inference Time** | **1.10 ms** | $< 3.0 \text{ ms}$ | **PASSED** |
| **Median Inference Time**| **0.98 ms** | $< 3.0 \text{ ms}$ | **PASSED** |
| **95th Percentile** | **1.65 ms** | $< 5.0 \text{ ms}$ | **PASSED** |
| **End-to-End Loop Rate** | **30–60 FPS** | $\ge 30 \text{ FPS}$ | **PASSED** |

### Classification Accuracy
- **Validation Accuracy**: **98.75%**
- **Confusion Matrix**:
  ```
               Predicted Rock  Predicted Paper  Predicted Scissors  Predicted Background
  True Rock          35               2                  0                    0
  True Paper          0              43                  0                    0
  True Scissors       0               0                 35                    0
  True Background     0               0                  0                   45
  ```

---

## 📜 Acknowledgements & References

- Based on the neuromorphic event-camera research by **Sensors Group, INI (Institute of Neuroinformatics, UZH-ETH Zurich)**: [SensorsINI/dextra-roshambo-python](https://github.com/SensorsINI/dextra-roshambo-python).
- Original Authors: Tobi Delbruck et al.
