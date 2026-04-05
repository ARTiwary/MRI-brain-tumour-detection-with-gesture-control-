import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

# Import your model class
try:
    from .model import CNN 
except ImportError:
    from model import CNN

app = FastAPI(title="Brain Tumor Detection API")

# --- CORS SETUP ---
# Allows your index.html to talk to this python server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DEVICE & MODEL LOADING ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path Logic: Finds saved_models inside the app folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "saved_models", "tumor-detection-model.pth")

model = CNN()

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"✅ SUCCESS: Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ ERROR: Model structure mismatch: {e}")
else:
    print(f"❌ ERROR: Model file NOT FOUND at {MODEL_PATH}")

# CLASS NAMES (Alphabetical order is best practice for consistency)
CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# --- HELPER FUNCTIONS ---
def preprocess_image(image_bytes):
    # Load and ensure RGB
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Resize to match training
    img = img.resize((128, 128))
    # Normalize to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    # HWC to CHW
    img_array = np.transpose(img_array, (2, 0, 1))
    # Add batch dim and move to device
    return torch.from_numpy(img_array).unsqueeze(0).to(device)

# --- ROUTES ---
@app.get("/")
def home():
    return {"status": "Online", "docs": "/docs"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid format")

    try:
        content = await file.read()
        input_tensor = preprocess_image(content)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            conf, pred_idx = torch.max(probabilities, 1)

        return {
            "tumor_type": CLASS_NAMES[pred_idx.item()],
            "confidence": round(conf.item() * 100, 2),
            "all_scores": {
                CLASS_NAMES[i]: round(prob * 100, 2) 
                for i, prob in enumerate(probabilities[0].tolist())
            }
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)