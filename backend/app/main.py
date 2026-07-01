# main.py
import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import base64

try:
    from .model import CNN, ResNetModel, EfficientNetModel
except ImportError:
    from model import CNN, ResNetModel, EfficientNetModel

# ─────────────────────────────────────────
# APP
# ─────────────────────────────────────────
app = FastAPI(title="Brain Tumor Detection API | Neuro-Core v4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ─────────────────────────────────────────
# NORMALIZATION STATS
# ─────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ⚠ REQUIRED: replace these with the actual values printed by your training
# notebook's Section 3 (`compute_dataset_stats(tr_imgs)` -> "CNN dataset
# mean / std"). The CNN was trained with stats computed from YOUR dataset,
# not ImageNet — using the wrong stats here will quietly degrade CNN
# predictions even though the server runs without errors.
CNN_MEAN = [0.000, 0.000, 0.000]   # <-- PASTE real printed value here
CNN_STD  = [1.000, 1.000, 1.000]   # <-- PASTE real printed value here

# ─────────────────────────────────────────
# MODEL REGISTRY
# Each model carries its OWN input size + normalization, matching exactly
# what it was trained with in the notebook. This is the part that was
# wrong before (everything was forced through 128×128 + ImageNet stats).
# ─────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

AVAILABLE_MODELS = {
    "cnn": {
        "label"      : "Custom CNN (128×128) — Test acc: 89.94%",
        "path"       : os.path.join(CURRENT_DIR, "saved_models", "tumor-detection-cnn-128.pth"),
        "class"      : CNN,
        "kwargs"     : {"num_classes": 4},
        "input_size" : 128,
        "mean"       : CNN_MEAN,
        "std"        : CNN_STD,
        "wraps_model": False,   # state_dict keys are NOT nested under "model."
    },
    "resnet18": {
        "label"      : "ResNet18 (224×224, transfer learning) — Test acc: 95.12%",
        "path"       : os.path.join(CURRENT_DIR, "saved_models", "tumor-detection-resnet18-224.pth"),
        "class"      : ResNetModel,
        "kwargs"     : {"num_classes": 4},
        "input_size" : 224,
        "mean"       : IMAGENET_MEAN,
        "std"        : IMAGENET_STD,
        "wraps_model": True,    # ResNetModel wraps backbone under self.model
    },
    "efficientnet_b0": {
        "label"      : "EfficientNet-B0 (224×224, transfer learning) — Test acc: 95.75%",
        "path"       : os.path.join(CURRENT_DIR, "saved_models", "tumor-detection-efficientnet-b0-224.pth"),
        "class"      : EfficientNetModel,
        "kwargs"     : {"num_classes": 4},
        "input_size" : 224,
        "mean"       : IMAGENET_MEAN,
        "std"        : IMAGENET_STD,
        "wraps_model": True,    # EfficientNetModel wraps backbone under self.model
    },
}

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# Cache — models stay in memory after first load
_model_cache: dict = {}

# ─────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────
def load_model(model_key: str):
    if model_key in _model_cache:
        return _model_cache[model_key]

    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_key}. "
                         f"Choose from: {list(AVAILABLE_MODELS.keys())}")

    cfg   = AVAILABLE_MODELS[model_key]
    model = cfg["class"](**cfg["kwargs"])

    if not os.path.exists(cfg["path"]):
        raise FileNotFoundError(
            f"Weights not found: {cfg['path']}. "
            f"Did you train and save this model, and does the filename match "
            f"exactly what Section 8/16/22D's `*_ckpt` saved to?"
        )

    checkpoint = torch.load(cfg["path"], map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    # ── Fix key nesting mismatch for wrapped models ──
    # ResNetModel / EfficientNetModel store the backbone under self.model,
    # but checkpoints saved straight from the notebook's bare torchvision
    # model have flat keys (e.g. "conv1.weight", "features.0.weight").
    if cfg["wraps_model"]:
        first_key = next(iter(state_dict.keys()))
        if not first_key.startswith("model."):
            state_dict = {f"model.{k}": v for k, v in state_dict.items()}
            print(f"  \u2139 Remapped {model_key} keys \u2192 model.*")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    meta = {}
    if isinstance(checkpoint, dict):
        meta = {
            "val_acc"  : checkpoint.get("val_acc",  "?"),
            "epoch"    : checkpoint.get("epoch",    "?"),
            "val_loss" : checkpoint.get("val_loss", "?"),
        }

    print(f"\u2705 Loaded [{model_key}] — {meta}")
    _model_cache[model_key] = model
    return model

# Pre-load all models at startup
for _key in AVAILABLE_MODELS:
    try:
        load_model(_key)
    except Exception as _e:
        print(f"\u26a0 Could not load [{_key}]: {_e}")

# ─────────────────────────────────────────
# PREPROCESSING — built per-model, matches training pipeline exactly
# ─────────────────────────────────────────
def get_preprocess(model_key: str):
    cfg = AVAILABLE_MODELS[model_key]
    return transforms.Compose([
        transforms.Resize((cfg["input_size"], cfg["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])


def preprocess_image(image_bytes: bytes, model_key: str):
    img      = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original = img.copy()
    preprocess = get_preprocess(model_key)
    tensor = preprocess(img).unsqueeze(0).to(device)
    return tensor, original

# ─────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────
def generate_gradcam(
    input_tensor : torch.Tensor,
    model        : nn.Module,
    original_img : Image.Image,
    pred_idx     : int,
    output_size  : int,
) -> str:
    """Returns base64-encoded JPEG of Grad-CAM overlay, sized to match the
    model's own input resolution (128 for CNN, 224 for ResNet18/EfficientNet)."""

    activations, gradients = [], []

    target_layer = model.last_conv_layer   # defined per-model in model.py

    h_act = target_layer.register_forward_hook(
        lambda m, i, o: activations.append(o)
    )
    h_grad = target_layer.register_full_backward_hook(
        lambda m, gi, go: gradients.append(go[0])
    )

    model.zero_grad()
    out = model(input_tensor)
    out[0, pred_idx].backward()

    h_act.remove()
    h_grad.remove()

    weights = torch.mean(gradients[0], dim=(2, 3), keepdim=True)
    cam     = torch.sum(weights * activations[0], dim=1, keepdim=True)
    cam     = F.relu(cam)
    cam     = cam - cam.min()
    cam     = cam / (cam.max() + 1e-8)
    cam     = cam.detach().cpu().numpy()[0, 0]   # (H, W)

    heatmap       = cv2.resize(cam, (output_size, output_size))
    heatmap       = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    orig_cv = cv2.cvtColor(
        np.array(original_img.resize((output_size, output_size))),
        cv2.COLOR_RGB2BGR
    )
    result = cv2.addWeighted(orig_cv, 0.6, heatmap_color, 0.4, 0)

    _, buffer = cv2.imencode(".jpg", result)
    return base64.b64encode(buffer).decode("utf-8")

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.get("/")
def home():
    return {
        "status"  : "Online",
        "model"   : "Brain Tumor Detection API",
        "device"  : str(device),
        "classes" : CLASS_NAMES,
        "docs"    : "/docs",
    }

@app.get("/health")
def health():
    return {
        "status"        : "healthy",
        "device"        : str(device),
        "models_loaded" : list(_model_cache.keys()),
    }

@app.get("/models")
def list_models():
    """Returns available models for frontend dropdown."""
    return {
        "models": [
            {
                "key"        : key,
                "label"      : cfg["label"],
                "input_size" : cfg["input_size"],
                "loaded"     : key in _model_cache,
            }
            for key, cfg in AVAILABLE_MODELS.items()
        ],
        "default": "efficientnet_b0",   # best accuracy + smallest model — see model_comparison results
    }

@app.post("/predict")
async def predict(
    file      : UploadFile = File(...),
    model_key : str        = Query(default="efficientnet_b0",
                                   description="Model to use: 'cnn', 'resnet18', or 'efficientnet_b0'"),
):
    """
    Predict brain tumor type from MRI image.

    - **file**: MRI image (JPEG or PNG)
    - **model_key**: which model to use (`cnn`, `resnet18`, or `efficientnet_b0`)

    Returns tumor type, confidence, per-class scores, and Grad-CAM heatmap.

    NOTE (clinical caveat): Glioma sensitivity for all three models is below
    the project's 95% target benchmark (CNN 80.7%, ResNet18 83.8%,
    EfficientNet-B0 85.8% on held-out test data, see model_comparison
    results). This API is a research/demo artifact, not a validated
    diagnostic tool — do not present "no tumor detected" or low-confidence
    Glioma scores as ruling out disease.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Upload a JPEG or PNG image."
        )

    if model_key not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_key '{model_key}'. "
                   f"Choose from: {list(AVAILABLE_MODELS.keys())}"
        )

    try:
        model = load_model(model_key)
        cfg = AVAILABLE_MODELS[model_key]

        content                = await file.read()
        input_tensor, orig_img = preprocess_image(content, model_key)

        model.eval()
        with torch.no_grad():
            output         = model(input_tensor)
            probabilities  = F.softmax(output, dim=1)
            conf, pred_idx = torch.max(probabilities, 1)

        pred_class = CLASS_NAMES[pred_idx.item()]
        confidence = round(conf.item() * 100, 2)
        all_scores = {
            CLASS_NAMES[i]: round(p * 100, 2)
            for i, p in enumerate(probabilities[0].tolist())
        }

        cam_tensor  = input_tensor.clone().requires_grad_(True)
        heatmap_b64 = generate_gradcam(
            cam_tensor, model, orig_img, pred_idx.item(),
            output_size=cfg["input_size"],
        )

        return {
            "model_used" : cfg["label"],
            "tumor_type" : pred_class,
            "confidence" : confidence,
            "all_scores" : all_scores,
            "heatmap"    : heatmap_b64,
            "device"     : str(device),
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)