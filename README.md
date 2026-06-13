MRI Brain Tumor Detection with Gesture Control (for contact less interaction for doctors ti maintain sterilized environmental)
1.
<div align="center">

# 🧠 NEURO-CORE 
### Brain Tumor MRI Detection with Gesture Control

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/License-MIT-00f2ff?style=for-the-badge)](LICENSE)

> A sterile, gesture-controlled neural diagnostic interface for brain tumor classification from MRI scans — powered by a custom CNN and a fine-tuned ResNet18 backbone, with real Grad-CAM explainability and hand-tracking control.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Models](#-models)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Contributing](#-contributing)

---

## 🔬 Overview

NEURO-CORE is an end-to-end brain tumor detection system built as a college project. It classifies brain MRI scans into four categories using deep learning, provides Grad-CAM heatmaps for explainability, and features a unique gesture-controlled interface powered by MediaPipe hand tracking — allowing users to interact with the diagnostic system without touching a keyboard or mouse.

**Clinical context:** Early and accurate brain tumor detection is critical. This system targets four of the most common brain tumor types found in MRI datasets, achieving **94.81% test accuracy** with the ResNet18 model.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Dual Model Support** | Switch between Custom CNN (92%) and ResNet18 (94.81%) via dropdown |
| **Grad-CAM Heatmaps** | Backend-generated TURBO colormap activation maps showing where the model looks |
| **Gesture Control** | Full hand-tracking interface — pinch to click, dwell to select, two-hand scroll |
| **Brain MRI Validation** | 5-layer filter rejects non-MRI images before inference |
| **Batch Processing** | Upload and process multiple scans in one sweep |
| **Folder Browser** | Drag-and-drop file tree with gesture support |
| **Fullscreen Viewer** | Water ripple zoom effect on scan images |
| **3D Sphere** | Hand-controlled morphing Three.js sphere when gesture mode is active |
| **FastAPI Backend** | REST API with `/predict`, `/models`, `/health` endpoints |

---

## 🎬 Demo

### Gesture Controls

| Gesture | Action |
|---|---|
| 👌 Pinch (index + thumb) | Click / Select |
| ⏱️ Hover 1.4s | Dwell click |
| ✋ Palm flash | Visual reset |
| ✊ Air grab × 3 | Turn gesture OFF |
| 🤲 Two hands (modal) | Scroll result modal |
| 🤏 Pinch in viewer | Zoom in |
| ✋ Open hand in viewer | Zoom out |
| 👋 Palm 3s in viewer | Close viewer |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FRONTEND (Browser)                 │
│  index.html + script.js                             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ MediaPipe   │  │  Three.js    │  │  File Tree │ │
│  │ Hand Track  │  │  3D Sphere   │  │  Browser   │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
│  ┌─────────────────────────────────────────────────┐│
│  │         Gesture Engine v5.0                     ││
│  │  Pinch · Dwell · Drag · Scroll · Air Grab       ││
│  └─────────────────────────────────────────────────┘│
└──────────────────────┬──────────────────────────────┘
                       │ HTTP POST /predict
                       ▼
┌─────────────────────────────────────────────────────┐
│                BACKEND (FastAPI)                     │
│  main.py                                            │
│  ┌──────────────────────────────────────────────┐  │
│  │  Brain MRI Validation (5 checks)             │  │
│  │  Aspect ratio · Grayscale · Contrast ·       │  │
│  │  Dark border · Minimum size                  │  │
│  └──────────────────┬───────────────────────────┘  │
│                     ▼                               │
│  ┌──────────────────────────────────────────────┐  │
│  │  Model Inference                             │  │
│  │  ┌────────────┐    ┌────────────────────┐   │  │
│  │  │ Custom CNN │ OR │ ResNet18 Pretrained │   │  │
│  │  │  92% acc   │    │    94.81% acc       │   │  │
│  │  └────────────┘    └────────────────────┘   │  │
│  └──────────────────┬───────────────────────────┘  │
│                     ▼                               │
│  ┌──────────────────────────────────────────────┐  │
│  │  Grad-CAM (TURBO colormap)                   │  │
│  │  Threshold mask · Gaussian blur · Colorbar   │  │
│  └──────────────────────────────────────────────┘  │
│                     │                               │
│         Returns JSON with heatmap (base64)          │
└─────────────────────────────────────────────────────┘
```

---

## 🤖 Models

### Model 1 — Custom CNN

A 4-block convolutional network trained from scratch on the brain tumor MRI dataset.

```
Input (3×128×128)
  → Block 1: Conv2d(3→32)  + BN + ReLU + MaxPool + Dropout2d(0.1)   → 64×64
  → Block 2: Conv2d(32→64) + BN + ReLU + MaxPool + Dropout2d(0.1)   → 32×32
  → Block 3: Conv2d(64→128)+ BN + ReLU + MaxPool + Dropout2d(0.2)   → 16×16
  → Block 4: Conv2d(128→256)+BN + ReLU + MaxPool + Dropout2d(0.2)   → 8×8
  → Flatten → Linear(16384→512) → ReLU → Dropout(0.5)
           → Linear(512→128)   → ReLU → Dropout(0.3)
           → Linear(128→4)
```

| Metric | Value |
|---|---|
| Parameters | ~8.8M |
| Test Accuracy | 92.00% |
| Val Accuracy | 96.70% |
| Epochs | 50 |
| Optimizer | Adam (lr=0.001, weight_decay=1e-4) |
| Scheduler | StepLR (step=15, gamma=0.5) |

### Model 2 — ResNet18 (Pretrained)

ImageNet-pretrained ResNet18 with a custom classifier head, fine-tuned with class weights to address Glioma class imbalance.

```
ResNet18 backbone (pretrained on ImageNet)
  → layer4[-1].conv2  ← Grad-CAM target
  → AdaptiveAvgPool
  → fc: Linear(512→256) → ReLU → Dropout(0.4) → Linear(256→4)
```

| Metric | Value |
|---|---|
| Parameters | ~11.7M |
| Test Accuracy | 94.81% |
| Val Accuracy | 97.32% |
| Epochs | 60 (early stopping at 58) |
| Backbone LR | 1e-4 |
| Head LR | 1e-3 |
| Class weights | Glioma×5, Meningioma×2 |

### Per-class Results (ResNet18)

| Class | Sensitivity | Specificity | Support |
|---|---|---|---|
| Glioma | 83.00% | 99.58% | 400 |
| Meningioma | 97.50% ✓ | 96.08% ✓ | 400 |
| Pituitary | 98.75% ✓ | 99.58% ✓ | 400 |
| No Tumor | 100.00% ✓ | 97.83% ✓ | 400 |
| **Macro avg** | **94.81%** | **98.27%** | **1600** |

> Target: Sensitivity ≥ 95%, Specificity ≥ 90%

---

## 📊 Dataset

**Brain Tumor MRI Dataset** — [Kaggle by Masoud Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

| Split | Glioma | Meningioma | Pituitary | No Tumor | Total |
|---|---|---|---|---|---|
| Training | ~1,321 | ~1,339 | ~1,457 | ~1,595 | ~5,712 |
| Testing | 400 | 400 | 400 | 400 | 1,600 |

**Preprocessing:**
- Resize to 128×128
- BGR → RGB conversion
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Augmentation (training only):**
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness=0.2, contrast=0.2)

---

## 📁 Project Structure

```
college project/
├── backend/
│   └── app/
│       ├── main.py                  # FastAPI app, Grad-CAM, validation, routes
│       ├── model.py                 # CNN and ResNetModel class definitions
│       └── saved_models/
│           ├── tumor-detection-model.pth        # Custom CNN weights
│           └── tumor-detection-resnet18.pth     # ResNet18 weights
│
├── frontend/
│   ├── index.html                   # Full UI — gesture controls, cards, modal
│   └── script.js                    # Gesture engine, folder browser, API calls
│
├── MRI/
│   └── Class-MRI-Brain-Tumor-Detector.ipynb    # Training notebook
│
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- Anaconda or pip
- CUDA-capable GPU (optional, CPU works)
- Node.js not required — frontend is plain HTML/JS

### 1. Clone the repo

```bash
git clone https://github.com/ARTiwary/MRI-brain-tumour-detection-with-gesture-control-.git
cd MRI-brain-tumour-detection-with-gesture-control-
```

### 2. Create environment

```bash
conda create -n mri_project python=3.10
conda activate mri_project
```

### 3. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install fastapi uvicorn python-multipart pillow opencv-python numpy
```

### 4. Add model weights

Place your trained `.pth` files in:
```
backend/app/saved_models/
  tumor-detection-model.pth
  tumor-detection-resnet18.pth
```

> **Note:** Weights are not included in this repo due to file size. Train using the notebook or contact the author.

---

## 🚀 Usage

### Start the backend

```powershell
cd backend/app
uvicorn main:app --reload
```

Backend runs at `http://127.0.0.1:8000`

### Open the frontend

Open `frontend/index.html` directly in your browser — no server needed.

### Using the interface

1. **Load scans** — click FOLDER or FILES in the left panel, or drag into the drop zone
2. **Select model** — choose Custom CNN or ResNet18 from the dropdown
3. **Process** — click `INITIALIZE NEURAL SWEEP`
4. **View results** — tumor type, confidence, Grad-CAM heatmap, probability bars
5. **Enable gestures** — click `GESTURE OFF` button to activate hand tracking

---

## 📡 API Reference

### `GET /`
Returns API status and available classes.

### `GET /health`
Returns server health and loaded models.

### `GET /models`
Returns available models for the frontend dropdown.

```json
{
  "models": [
    {"key": "cnn",      "label": "Custom CNN — 92.00%",   "loaded": true},
    {"key": "resnet18", "label": "ResNet18 — 94.81%",      "loaded": true}
  ],
  "default": "resnet18"
}
```

### `POST /predict`

**Query params:** `model_key` = `cnn` or `resnet18` (default: `resnet18`)

**Body:** `multipart/form-data` with `file` (JPEG or PNG)

**Response:**
```json
{
  "model_used" : "ResNet18 Pretrained — Test acc: 94.81%",
  "tumor_type" : "Glioma",
  "confidence" : 91.34,
  "all_scores" : {
    "Glioma"     : 91.34,
    "Meningioma" : 5.21,
    "Pituitary"  : 2.10,
    "No Tumor"   : 1.35
  },
  "heatmap" : "<base64-encoded JPEG>",
  "device"  : "cuda"
}
```

**Validation errors (422):**
```json
{
  "detail": {
    "error"   : "NOT_BRAIN_MRI",
    "message" : "Image appears to be a color photograph (color variance: 42.3).",
    "hint"    : "Please upload a grayscale brain MRI scan (T1/T2/FLAIR)."
  }
}
```

---

## 📈 Results

### Training Curves

```
Epoch 01/50 | Train Loss: 1.2677 Acc: 0.4779 | Val Loss: 0.8845 Acc: 0.6062
Epoch 10/50 | Train Loss: 0.5181 Acc: 0.8013 | Val Loss: 0.4049 Acc: 0.8473
Epoch 25/50 | Train Loss: 0.2746 Acc: 0.8969 | Val Loss: 0.2095 Acc: 0.9304
Epoch 50/50 | Train Loss: 0.1131 Acc: 0.9600 | Val Loss: 0.1032 Acc: 0.9670
```

### Overfitting Analysis

| Metric | Value | Status |
|---|---|---|
| Train → Val gap | 0.7% | ✅ Healthy |
| Loss gap (epoch 50) | 0.007 | ✅ Healthy |
| Val loss trend | Declining | ✅ No overfit |
| LR scheduler kick (epoch 31) | Sharp improvement | ✅ Expected |

### Brain MRI Validation — 5 Checks

| Check | Catches |
|---|---|
| Aspect ratio (0.6–1.7) | Landscape photos, banners, portraits |
| Minimum size (64×64) | Thumbnails, icons |
| Grayscale (R≈G≈B, diff < 30) | Color photographs, screenshots |
| Contrast (std dev > 15) | Blank images, solid fills |
| Dark border (mean < 180) | Natural photos with bright backgrounds |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch 2.0+, torchvision |
| Models | Custom CNN, ResNet18 (pretrained) |
| Explainability | Grad-CAM with TURBO colormap |
| Backend | FastAPI, Uvicorn |
| Image Processing | OpenCV, Pillow, NumPy |
| Hand Tracking | MediaPipe Hands |
| 3D Graphics | Three.js r128 |
| Frontend | Vanilla HTML/CSS/JS |
| Fonts | Orbitron, Rajdhani, Space Mono |
| Data Analysis | scikit-learn, matplotlib, seaborn |
| Training | CUDA GPU acceleration |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

```bash
git checkout -b feature/your-feature
git commit -m "feat: add your feature"
git push origin feature/your-feature
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built by [ARTiwary](https://github.com/ARTiwary)**

*College Project — Brain Tumor Detection with Gesture Control*

</div>
