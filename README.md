<div align="center">

# 🧠 NeuroCore
### Brain Tumor MRI Detection with Gesture Control

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/License-MIT-2563eb?style=for-the-badge)](LICENSE)

> A sterile, gesture-controlled neural diagnostic interface for brain tumor classification from MRI scans — powered by three independently trained deep learning models (Custom CNN, ResNet-18, EfficientNet-B0), Grad-CAM explainability, LLM-generated plain-language summaries via Groq, and a fully in-browser MediaPipe hand-tracking interface designed for contactless clinical use.

**Research context:** This project demonstrates that gesture-controlled, AI-assisted medical imaging interfaces can be built without proprietary hardware. The gesture system requires no gloves, markers, or special cameras — only a standard webcam.

**Clinical caveat:** This is a research prototype, not a validated clinical tool. Glioma sensitivity across all three models is below the 95% benchmark target used in evaluation. Any diagnostic decision must involve a qualified radiologist.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Gesture Controls](#-gesture-controls)
- [Architecture](#-architecture)
- [Models](#-models)
- [Medical-Grade Evaluation](#-medical-grade-evaluation)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Tech Stack](#-tech-stack)

---

## 🔬 Overview

NeuroCore is an end-to-end brain tumor MRI classification system built as a B.Tech CSE research project. It classifies brain MRI scans into four categories — Glioma, Meningioma, Pituitary tumor, and No Tumor — using three independently trained deep learning models, provides Grad-CAM activation heatmaps for explainability, and features a gesture-controlled interface powered by MediaPipe Hands that allows radiologists to navigate and inspect scans without physically touching any input device.

**Why gesture control?** Operating radiology workstations requires repeatedly breaking sterile gloves to use a keyboard or mouse. A contactless gesture interface addresses this directly.

**Best model accuracy:** EfficientNet-B0 — **95.75% test accuracy** with the smallest parameter count of the three models (~5.3M vs 8.8M CNN / 11.3M ResNet-18).

---

## ✨ Features

| Feature | Description |
|---|---|
| **Three-Model Support** | Custom CNN (89.94%), ResNet-18 (95.12%), EfficientNet-B0 (95.75%) — selectable via dropdown |
| **Grad-CAM Heatmaps** | Per-model backend-generated activation maps using each model's validated target layer |
| **Medical-Grade Evaluation** | Validation-tuned cost-weighted decision rule, bootstrap 95% CIs, per-class sensitivity/specificity |
| **Gesture Control** | Full in-browser hand tracking — pinch, dwell, drag, two-hand scroll, pinch zoom, air-grab disable |
| **Brain MRI Filter** | Low-confidence rejection rejects non-MRI images before displaying a result |
| **Batch Processing** | Upload and process any number of scans in one sweep, results rendered as they arrive |
| **Folder Browser** | Hierarchical file tree with drag-and-drop and full gesture support |
| **Fullscreen Viewer** | Water-ripple zoom effect on scan images, gesture-controlled zoom in/out |
| **3D Sphere** | Hand-controlled morphing Three.js particle sphere when gesture mode is active |
| **FastAPI Backend** | Single `/predict` endpoint with model-key param, in-memory model cache, per-model preprocessing |

---

## 🎬 Gesture Controls

**Library:** MediaPipe Hands (`@mediapipe/hands` + `@mediapipe/camera_utils`) — runs entirely in-browser via WebAssembly. No server call, no OpenCV, no TensorFlow.js.

**Cursor:** Index fingertip (landmark 8), smoothed with exponential moving average (α = 0.16).

| Gesture | Detection | Action |
|---|---|---|
| Index fingertip movement | Landmark 8, EMA smoothed | Move cursor |
| Pinch — index + thumb < 0.047 dist | Euclidean dist landmarks 4↔8 | Click / select |
| Hover 1.4 s without pinching | Dwell timer on `.g-clickable` | Dwell click (ring progress fills) |
| Pinch hold on file + release over drop zone | Drag state machine | Drag file to drop zone |
| Pinch hold on file + release away | Drag state machine, miss path | Cancel drag |
| Two hands (results modal open) | `allHands.length === 2` | Two-hand scroll with momentum |
| Two hands spread/squeeze (viewer open) | Distance between both hand-8 landmarks | Pinch-zoom in fullscreen viewer |
| Single pinch close (viewer open) | Landmark 4↔8 < 0.047 | Zoom in |
| Open hand sustained (viewer open) | Landmark 4↔8 > 0.12 | Zoom out |
| Palm hold 3 s (viewer open) | All fingertips above PIP joints | Close fullscreen viewer |
| Palm flash (main screen) | Same palm check, 3 s gap | Visual flash + reminder |
| Air-pinch × 3 in empty space | `grabWasOnFile = false` flag + counter | Turn gesture mode OFF |

**Thresholds:** Pinch closed < 0.047 normalised units · Pinch open > 0.085 · Dead zone between prevents jitter · Grab counter resets after 2 s gap.

---

## 🏗️ Architecture

**One line:** A single `POST /predict?model_key={key}` endpoint — the frontend passes whichever model the user selected, the backend loads the corresponding checkpoint from an in-memory cache, runs per-model preprocessing, runs inference, generates a Grad-CAM heatmap against that model's registered target layer, optionally calls Groq for a plain-language summary, and returns a single JSON response.

```
Browser (HTML + CSS + Vanilla JS)
│
├── MediaPipe Hands (WebAssembly, in-browser, no server)
│     └── gesture events → cursor / click / drag / scroll / zoom
│
├── Three.js morphing sphere (gesture ON background)
│
├── Folder browser + drop zone queue → FormData (image bytes)
│
└── fetch POST /predict?model_key={key}&explain={bool}
      │
      ▼
FastAPI  (uvicorn, http://127.0.0.1:8000)
│
├── GET  /models   → model registry + loaded status
├── GET  /health   → loaded models + device
└── POST /predict  ← image bytes + model_key + explain flag
      │
      ├── load_model(key)              ← _model_cache dict (in-memory)
      │     └── reads .pth once        ← saved_models/ folder
      │
      ├── preprocess_image()
      │     ├── CNN:          128×128  + dataset-computed mean/std
      │     └── ResNet / Eff: 224×224  + ImageNet mean/std
      │
      ├── model.forward()              ← CNN / ResNet-18 / EfficientNet-B0
      │     └── softmax → class scores
      │
      ├── generate_gradcam()
      │     ├── CNN:       model.features[-last Conv2d]
      │     ├── ResNet-18: model.layer4[-1]   (post-residual-addition)
      │     └── EfficientNet-B0: model.features[-1]
      │     └── OpenCV JET overlay → base64 JPEG
      │
      │
      └── JSON response
            { tumor_type, confidence, all_scores, heatmap, explanation? }
```

---

## 🤖 Models

All three models were trained on an identical, fixed, seeded train/val split (80/20 of the training partition) so results are directly comparable. All use Focal Loss with class weights (Glioma×4, Meningioma×1.5) to address the class-sensitivity imbalance.

---

### Model 1 — Custom CNN (trained from scratch)

```
Input (3 × 128 × 128)
  → Block 1: Conv2d(3→32,  k3) + BN + ReLU + MaxPool2d + Dropout2d(0.1) → 64×64
  → Block 2: Conv2d(32→64, k3) + BN + ReLU + MaxPool2d + Dropout2d(0.1) → 32×32
  → Block 3: Conv2d(64→128,k3) + BN + ReLU + MaxPool2d + Dropout2d(0.2) → 16×16
  → Block 4: Conv2d(128→256,k3)+ BN + ReLU + MaxPool2d + Dropout2d(0.2) → 8×8
  → Flatten(16384) → Linear(16384→512) → ReLU → Dropout(0.5)
             → Linear(512→128) → ReLU → Dropout(0.3)
             → Linear(128→4)
```

| Metric | Value |
|---|---|
| Total parameters | ~8.8 M |
| Test accuracy | 89.94% |
| Val accuracy | 93.84% |
| Train accuracy | 95.11% |
| Train → val gap | 1.27% ✅ healthy |
| Loss function | Focal Loss (γ=2, class weights) |
| Optimizer | Adam lr=1e-3, weight_decay=1e-4 |
| Scheduler | StepLR (step=15, γ=0.5) |
| Max epochs | 50 |
| Normalization | Dataset-computed mean/std |
| Grad-CAM target | Last Conv2d in `features` block |

---

### Model 2 — ResNet-18 (transfer learning)

```
ResNet-18 backbone (ImageNet pretrained)
  → layer1 → layer2 → layer3 → layer4[-1]  ← Grad-CAM target
  → AdaptiveAvgPool2d
  → classifier head: Linear(512→256) → ReLU → Dropout(0.4) → Linear(256→4)
```

| Metric | Value |
|---|---|
| Total parameters | ~11.3 M |
| Test accuracy | 95.12% |
| Val accuracy | 99.02% |
| Train accuracy | 99.82% |
| Train → val gap | 0.80% ✅ healthy |
| Loss function | Focal Loss (γ=2, class weights) |
| Backbone LR | 1e-4 (discriminative) |
| Head LR | 1e-3 |
| Scheduler | StepLR (step=10, γ=0.5) |
| Max epochs | 60 (early stopping patience=10) |
| Normalization | ImageNet mean/std |
| Grad-CAM target | `layer4[-1]` (post-residual-addition block) |

---

### Model 3 — EfficientNet-B0 (transfer learning) ⭐ Best

```
EfficientNet-B0 backbone (ImageNet pretrained)
  → features[0] stem → features[1..7] MBConv stages → features[8]  ← Grad-CAM target
  → AdaptiveAvgPool2d
  → classifier head: Dropout(0.3) → Linear(1280→256) → ReLU → Dropout(0.3) → Linear(256→4)
```

| Metric | Value |
|---|---|
| Total parameters | ~5.3 M |
| Test accuracy | 95.75% ⭐ |
| Val accuracy | 99.11% |
| Train accuracy | 99.64% |
| Train → val gap | 0.53% ✅ healthy |
| Loss function | Focal Loss (γ=2, class weights) |
| Backbone LR | 1e-4 (discriminative) |
| Head LR | 1e-3 |
| Scheduler | StepLR (step=10, γ=0.5) |
| Max epochs | 60 (early stopping patience=10) |
| Normalization | ImageNet mean/std |
| Grad-CAM target | `features[-1]` (head conv, stage 8) |

> EfficientNet-B0 achieves the highest test accuracy with the fewest parameters of the three models — making it the preferred choice for both accuracy and deployment efficiency.

---

## 📊 Medical-Grade Evaluation

Evaluation follows a medical AI convention: thresholds are tuned on the **validation set only** and applied once to the test set — never tuned against test data. A cost-weighted decision rule (`argmax(prob × weight)`) is used instead of raw probability thresholding, with a safety floor that prevents improving one class's sensitivity at the cost of another class dropping below 80%.

Confidence intervals are bootstrap 95% CIs (1,000 resamples).

### Per-class Sensitivity / Specificity — Test Set

| Class | CNN | CNN CI | ResNet-18 | ResNet-18 CI | EfficientNet-B0 | EfficientNet-B0 CI |
|---|---|---|---|---|---|---|
| Glioma Sens | 80.71% ✗ | [76.9, 84.4] | 83.75% ✗ | [80.3, 87.1] | 85.75% ✗ | [82.3, 89.0] |
| Glioma Spec | 97.33% ✓ | [96.4, 98.3] | 99.59% ✓ | [99.2, 99.9] | 99.76% ✓ | [99.5, 100.0] |
| Meningioma Sens | 80.70% ✗ | [76.8, 84.7] | 97.47% ✓ | [95.7, 98.8] | 98.28% ✓ | [96.9, 99.5] |
| Meningioma Spec | 96.27% ✓ | [95.2, 97.3] | 97.42% ✓ | [96.5, 98.3] | 96.58% ✓ | [95.6, 97.6] |
| Pituitary Sens | 99.00% ✓ | [98.0, 99.8] | 99.26% ✓ | [98.3, 100.0] | 99.03% ✓ | [97.9, 99.8] |
| Pituitary Spec | 97.99% ✓ | [97.2, 98.7] | 99.83% ✓ | [99.6, 100.0] | 99.67% ✓ | [99.3, 99.9] |
| No Tumor Sens | 99.27% ✓ | [98.4, 100.0] | 100.00% ✓ | [100.0, 100.0] | 100.00% ✓ | [100.0, 100.0] |
| No Tumor Spec | 94.97% ✓ | [93.7, 96.1] | 96.66% ✓ | [95.6, 97.6] | 98.34% ✓ | [97.6, 99.0] |

> **Target:** Sensitivity ≥ 95%, Specificity ≥ 90%

### Overall Comparison

| Metric | CNN | ResNet-18 | EfficientNet-B0 |
|---|---|---|---|
| Overall Accuracy | 89.94% | 95.12% | **95.75%** |
| Macro Sensitivity | 89.94% | 95.12% | **95.75%** |
| Macro Specificity | 96.64% | 98.38% | **98.59%** |
| Parameters | 8.8 M | 11.3 M | **5.3 M** |

### Glioma Cost-Weight Tradeoff (test set — informational)

Threshold/weight tuning was tested across all three architectures. Results confirm the Glioma sensitivity gap cannot be closed by post-hoc decision rule adjustment without unacceptable cost to Meningioma (CNN) or with negligible gain (ResNet-18, EfficientNet-B0). This is reported as a data/architecture limitation, not a tuning failure.

| Weight | CNN Glioma | CNN Menin. | ResNet Glioma | ResNet Menin. | Eff Glioma | Eff Menin. |
|---|---|---|---|---|---|---|
| 1.0 | 80.75% | 80.75% | 83.75% | 97.50% | 85.75% | 98.25% |
| 2.0 | 81.75% | 55.75% | 84.50% | 97.00% | 85.75% | 97.75% |
| 4.0 | 81.75% | 40.25% | 85.25% | 95.75% | 86.25% | 97.50% |

**Conclusion:** Glioma sensitivity is a data-volume limitation. The same gap appears independently across three architectures trained from identical splits, ruling out architecture-specific causes.

---

## 📊 Dataset

**Brain Tumor MRI Dataset** — [Kaggle by Masoud Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

| Split | Glioma | Meningioma | Pituitary | No Tumor | Total |
|---|---|---|---|---|---|
| Training (full) | 1,400 | 1,400 | 1,400 | 1,400 | 5,600 |
| Train partition (80%) | ~1,120 | ~1,120 | ~1,120 | ~1,120 | ~4,480 |
| Val partition (20%) | ~280 | ~280 | ~280 | ~280 | ~1,120 |
| Testing | 400 | 400 | 400 | 400 | 1,600 |

**Preprocessing:**

| Step | CNN | ResNet-18 / EfficientNet-B0 |
|---|---|---|
| Resize | 128 × 128 | 224 × 224 |
| Normalization | Dataset-computed mean/std | ImageNet mean/std |
| Color | BGR → RGB | BGR → RGB |

**Training augmentation (all models):**
- Random horizontal flip (p = 0.5) — anatomically valid for axial MRI
- Random rotation ±15° — patient head positioning variation
- Color jitter (brightness=0.2, contrast=0.2)
- No vertical flip — anatomically implausible for axial slices

**Known limitation:** The dataset carries no patient-identifier metadata, making it impossible to verify the absence of patient-level data leakage between training and test partitions (multiple slices from one patient in both splits). This is flagged in the paper's Limitations section.

---

## 📁 Project Structure

```
college project/
│
├── backend/
│   └── app/
│       ├── main.py                          # FastAPI app, per-model preprocessing,
│       │                                    # Grad-CAM, Groq explanation, routes
│       ├── model.py                         # CNN, ResNetModel, EfficientNetModel
│       │                                    # classes + AVAILABLE_MODELS registry
│       └── saved_models/
│           ├── tumor-detection-cnn-128.pth            # CNN weights
│           ├── tumor-detection-resnet18-224.pth        # ResNet-18 weights
│           └── tumor-detection-efficientnet-b0-224.pth # EfficientNet-B0 weights
│
├── frontend/
│   ├── index.html                           # Professional white UI — gesture controls,
│   │                                        # folder browser, result modal, viewer
│   └── script.js                            # Gesture engine v5.0, folder browser,
│                                            # batch processor, model selector,
│                                            # water-ripple viewer, Three.js sphere
│
├── MRI/
│   └── Class-MRI-Brain-Tumor-Detector.ipynb  # Full training pipeline:
│                                              # CNN → ResNet-18 → EfficientNet-B0
│                                              # + medical-grade evaluation
│                                              # + Grad-CAM + feature maps
│
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- pip or conda
- CUDA-capable GPU (optional — CPU works, slower inference)
- Node.js **not required** — frontend is plain HTML/JS

### 1. Clone the repository

```bash
git clone https://github.com/ARTiwary/MRI-brain-tumour-detection-with-gesture-control-.git
cd MRI-brain-tumour-detection-with-gesture-control-
```

### 2. Create environment

```bash
conda create -n neurocore python=3.10
conda activate neurocore
```

### 3. Install dependencies

```bash
# PyTorch — pick the right CUDA version for your GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Backend
pip install fastapi uvicorn python-multipart pillow opencv-python numpy groq

# Training / evaluation (optional — only needed to retrain)
pip install scikit-learn matplotlib seaborn
```

### 4. Add model weights

Place your trained `.pth` files in `backend/app/saved_models/`:

```
tumor-detection-cnn-128.pth
tumor-detection-resnet18-224.pth
tumor-detection-efficientnet-b0-224.pth
```

> **Note:** Weights are not included due to file size. Train using the notebook or contact the author.

If no key is set, the `/predict` endpoint works normally — the `explanation` field is simply absent from the response.

---

## 🚀 Usage

### Start the backend

```powershell
cd backend/app
uvicorn main:app --reload
```

Backend runs at `http://127.0.0.1:8000`. Swagger docs at `http://127.0.0.1:8000/docs`.

All three model checkpoints are loaded into memory at startup. First startup takes a few seconds; all subsequent inference calls are fast.

### Open the frontend

Open `frontend/index.html` directly in your browser — no build step, no server required.

### Workflow

1. **Load scans** — click Folder or Files in the left panel, or drag images into the drop zone
2. **Select model** — use the dropdown to switch between CNN, ResNet-18, or EfficientNet-B0
3. **Process** — click **Run neural diagnostic sweep**
4. **View results** — tumor class, confidence %, Grad-CAM heatmap, per-class probability bars
5. **Inspect** — click any result image to open the fullscreen water-ripple viewer
6. **Enable gestures** — click **Gesture off** button top-right to activate hand tracking

---

## 📡 API Reference

### `GET /`
Returns API status, device, and class names.

### `GET /health`
Returns server health, device, and list of currently loaded model keys

### `GET /models`
Returns the model registry for the frontend dropdown.

```json
{
  "models": [
    {"key": "cnn",              "label": "Custom CNN (128×128) — Test acc: 89.94%",            "input_size": 128,  "loaded": true},
    {"key": "resnet18",         "label": "ResNet18 (224×224, transfer learning) — Test acc: 95.12%", "input_size": 224, "loaded": true},
    {"key": "efficientnet_b0",  "label": "EfficientNet-B0 (224×224, transfer learning) — Test acc: 95.75%", "input_size": 224, "loaded": true}
  ],
  "default": "efficientnet_b0"
}
```

### `POST /predict`

**Query parameters:**

| Param | Type | Default | Values |
|---|---|---|---|
| `model_key` | string | `efficientnet_b0` | `cnn`, `resnet18`, `efficientnet_b0` |

**Body:** `multipart/form-data` with `file` field (JPEG or PNG).

**Response (200):**
```json
{
  "model_used"  : "EfficientNet-B0 (224×224, transfer learning) — Test acc: 95.75%",
  "tumor_type"  : "Glioma",
  "confidence"  : 91.34,
  "all_scores"  : {
    "Glioma"     : 91.34,
    "Meningioma" : 5.21,
    "Pituitary"  : 2.10,
    "No Tumor"   : 1.35
  },
  "heatmap"     : "<base64-encoded JPEG — JET colormap overlay>",
  "device"      : "cuda",
  "explanation" : "The model predicts Glioma with 91.3% confidence... [Groq, only if explain=true]"
}
```

**Error (400):** Invalid file type.
**Error (400):** Invalid `model_key`.
**Error (503):** Model weights file not found on disk.
**Error (500):** Inference or Grad-CAM failure.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Deep learning framework | PyTorch 2.0+, torchvision |
| Models | Custom CNN (scratch), ResNet-18 (pretrained), EfficientNet-B0 (pretrained) |
| Loss function | Focal Loss (γ=2) + class weighting |
| Explainability | Grad-CAM — per-model validated target layer |
| Backend | FastAPI, Uvicorn |
| Image processing | OpenCV, Pillow, NumPy |
| Gesture tracking | MediaPipe Hands (WebAssembly, in-browser) |
| 3D graphics | Three.js r128 |
| Frontend | Vanilla HTML / CSS / JavaScript (no framework, no build step) |
| Fonts | Inter, JetBrains Mono |
| Training analysis | scikit-learn, matplotlib, seaborn |
| Statistical evaluation | Bootstrap 95% CI (1,000 resamples), cost-weighted decision rule |
| Training hardware | CUDA GPU (NVIDIA) |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built by [ARTiwary](https://github.com/ARTiwary)**

*B.Tech CSE Research Project — Brain Tumor MRI Detection with Gesture Control*

*Research use only — not a validated clinical diagnostic tool*

</div>
