# ⚡ Ultra-Low Latency Invincible Rock-Paper-Scissors System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![Inference Latency](https://img.shields.io/badge/Inference_Latency-%3C0.60ms_(CPU)-brightgreen.svg)]()
[![Validation Accuracy](https://img.shields.io/badge/Val_Accuracy-93.92%25_(Real_Webcam)-success.svg)]()

An ultra-low latency, computer-vision-driven **Invincible Rock-Paper-Scissors** system designed to run on a standard laptop RGB webcam. 

Adapted from the neuromorphic Event Camera (DVS) project [SensorsINI/dextra-roshambo-python](https://github.com/SensorsINI/dextra-roshambo-python), this project replaces specialized neuromorphic event camera hardware with an optimized **Pseudo-Event Motion Masking Pipeline** (Frame Differencing + Sensitive Binary Thresholding), paired with a pure PyTorch **RoshamboNet v2 Tiny CNN** (with BatchNorm) and a **5-frame temporal majority filter**.

---

## 🎯 Key Features

- **Pseudo-Event Motion Pipeline**: Replaces neuromorphic DVS event streams with consecutive frame differencing ($|I_t - I_{t-1}|$) and calibrated binary thresholding (`threshold=18`) to isolate sharp, noise-free motion trails in real time.
- **Sub-1ms CPU Inference**: Powered by `RoshamboNet v2`, a lightweight 64×64 Tiny CNN ($115,172$ parameters with `BatchNorm2d` and `Dropout(0.1)`) that completes inference in **~0.60 ms on standard CPU** without requiring a GPU.
- **Motion Activity Curation**: Automated filtering (`min_gesture_pixels=15`) cleans zero-motion frames from gesture classes during bursts, ensuring clean label boundaries and preventing still-hand frames from confusing gestures with background.
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
  Binary Thresholding (cv2.threshold @ 18 -> Sharp Motion Trail)
                   │
                   ▼
  Center ROI Crop & Resize to 64x64 Grayscale Tensor [1, 1, 64, 64]
                   │
                   ▼
  PyTorch RoshamboNet v2 Tiny CNN with BatchNorm (0.60 ms CPU Inference)
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
├── model.py                 # RoshamboNet v2 (BatchNorm) & MajorityVote filter
├── pseudo_event_producer.py # Webcam motion differencing & 64x64 mask generation (thresh=18)
├── collect_data.py          # Interactive dataset generator with burst & motion filtering
├── train.py                 # Optimized training pipeline (AdamW, CosineAnnealing, Augmentations)
├── live_predict.py          # Real-time inference engine with HUD visualizer
├── motion_model.pth         # Trained model weights (93.92% real-world validation accuracy)
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
python3 collect_data.py --camera 0 --threshold 18 --burst_size 25
```
- Press **`r`** $\rightarrow$ Record 25-frame burst for **Rock**
- Press **`p`** $\rightarrow$ Record 25-frame burst for **Paper**
- Press **`s`** $\rightarrow$ Record 25-frame burst for **Scissors**
- Press **`b`** $\rightarrow$ Record 25-frame burst for **Background**
- Press **`q`** or **`ESC`** $\rightarrow$ Exit collection

*(Note: `--min_motion_pixels 15` automatically ignores blank frames before motion begins, ensuring high dataset quality.)*

---

### Step 2: Train the RoshamboNet Model
Train the 64×64 Tiny CNN on your dataset:
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

### Step 3: Run the Real-Time Invincible System
Launch the real-time webcam inference system:
```bash
python3 live_predict.py --camera 0 --fps 60 --model motion_model.pth --threshold 18
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

### CPU Inference Latency (Standard Laptop CPU)
| Metric | Result | Target Budget | Status |
| :--- | :--- | :--- | :--- |
| **Mean Inference Time** | **0.60 ms** | $< 3.0 \text{ ms}$ | **PASSED** |
| **Median Inference Time**| **0.57 ms** | $< 3.0 \text{ ms}$ | **PASSED** |
| **95th Percentile** | **0.84 ms** | $< 5.0 \text{ ms}$ | **PASSED** |
| **End-to-End Loop Rate** | **30–60 FPS** | $\ge 30 \text{ FPS}$ | **PASSED** |

### Real Webcam Validation Accuracy (3,459 Curated Samples)
- **Overall Validation Accuracy**: **93.92%**

#### Detailed Classification Report
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

#### Confusion Matrix
```
                 Predicted Rock  Predicted Paper  Predicted Scissors  Predicted Background
True Rock             174               4                 8                    4
True Paper              9             161                 3                    1
True Scissors           8               2               161                    1
True Background         0               0                 2                  153
```

---

## 📜 Acknowledgements & References

- Based on the neuromorphic event-camera research by **Sensors Group, INI (Institute of Neuroinformatics, UZH-ETH Zurich)**: [SensorsINI/dextra-roshambo-python](https://github.com/SensorsINI/dextra-roshambo-python).
- Original Authors: Tobi Delbruck et al.
