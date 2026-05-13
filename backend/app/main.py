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
    from .model import CNN, ResNetModel
except ImportError:
    from model import CNN, ResNetModel

# ─────────────────────────────────────────
# APP
# ─────────────────────────────────────────
app = FastAPI(title="Brain Tumor Detection API | Neuro-Core v3.4")

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
# MODEL REGISTRY
# ─────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

AVAILABLE_MODELS = {
    "cnn": {
        "label"  : "Custom CNN (4 blocks) — Test acc: 92.00%",
        "path"   : os.path.join(CURRENT_DIR, "saved_models",
                                "tumor-detection-model.pth"),
        "class"  : CNN,
        "kwargs" : {"num_classes": 4},
    },
    "resnet18": {
        "label"  : "ResNet18 Pretrained — Test acc: 94.81%",
        "path"   : os.path.join(CURRENT_DIR, "saved_models",
                                "tumor-detection-resnet18.pth"),
        "class"  : ResNetModel,
        "kwargs" : {"num_classes": 4},
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
            f"Weights not found: {cfg['path']}"
        )

    checkpoint = torch.load(cfg["path"], map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    # ── Fix ResNet18 key mismatch ──
    # Notebook saved keys as conv1.weight (flat resnet18)
    # ResNetModel wraps it under self.model → expects model.conv1.weight
    if model_key == "resnet18":
        first_key = next(iter(state_dict.keys()))
        if not first_key.startswith("model."):
            state_dict = {f"model.{k}": v for k, v in state_dict.items()}
            print("  ℹ Remapped ResNet18 keys → model.*")

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

    print(f"✅ Loaded [{model_key}] — {meta}")
    _model_cache[model_key] = model
    return model

# Pre-load both at startup
for _key in AVAILABLE_MODELS:
    try:
        load_model(_key)
    except Exception as _e:
        print(f"⚠ Could not load [{_key}]: {_e}")

# ─────────────────────────────────────────
# PREPROCESSING
# Matches training pipeline exactly
# ─────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image_bytes: bytes):
    img      = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original = img.copy()
    tensor   = preprocess(img).unsqueeze(0).to(device)   # (1,3,128,128)
    return tensor, original

# ─────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────
def generate_gradcam(
    input_tensor : torch.Tensor,
    model        : nn.Module,
    original_img : Image.Image,
    pred_idx     : int,
) -> str:
    """Returns base64-encoded JPEG of Grad-CAM overlay."""

    activations, gradients = [], []

    target_layer = model.last_conv_layer   # defined in model.py for both CNN + ResNet

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

    # Weighted combination of activations
    weights = torch.mean(gradients[0], dim=(2, 3), keepdim=True)
    cam     = torch.sum(weights * activations[0], dim=1, keepdim=True)
    cam     = F.relu(cam)
    cam     = cam - cam.min()
    cam     = cam / (cam.max() + 1e-8)
    cam     = cam.detach().cpu().numpy()[0, 0]   # (H, W)

    # Resize + colorize
    heatmap       = cv2.resize(cam, (128, 128))
    heatmap       = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay on original
    orig_cv = cv2.cvtColor(
        np.array(original_img.resize((128, 128))),
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
    """
    Returns available models for frontend dropdown.
    Frontend should call this on load to populate the selector.
    """
    return {
        "models": [
            {
                "key"   : key,
                "label" : cfg["label"],
                "loaded": key in _model_cache,
            }
            for key, cfg in AVAILABLE_MODELS.items()
        ],
        "default": "resnet18",
    }

@app.post("/predict")
async def predict(
    file      : UploadFile = File(...),
    model_key : str        = Query(default="resnet18",
                                   description="Model to use: 'cnn' or 'resnet18'"),
):
    """
    Predict brain tumor type from MRI image.

    - **file**: MRI image (JPEG or PNG)
    - **model_key**: which model to use (`cnn` or `resnet18`)

    Returns tumor type, confidence, per-class scores, and Grad-CAM heatmap.
    """
    # ── Validate file type ──
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Upload a JPEG or PNG image."
        )

    # ── Validate model key ──
    if model_key not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_key '{model_key}'. "
                   f"Choose from: {list(AVAILABLE_MODELS.keys())}"
        )

    try:
        model = load_model(model_key)

        content                = await file.read()
        input_tensor, orig_img = preprocess_image(content)

        # ── Prediction ──
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

        # ── Grad-CAM ──
        cam_tensor  = input_tensor.clone().requires_grad_(True)
        heatmap_b64 = generate_gradcam(
            cam_tensor, model, orig_img, pred_idx.item()
        )

        return {
            "model_used" : AVAILABLE_MODELS[model_key]["label"],
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