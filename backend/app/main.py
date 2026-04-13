import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
import base64

# Import your model class
try:
    from .model import CNN 
except ImportError:
    from model import CNN

app = FastAPI(title="Brain Tumor Detection API | Neuro-Core v3.4")

# --- CORS SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DEVICE & MODEL LOADING ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# --- HEATMAP (GRAD-CAM) GENERATOR ---
def generate_heatmap(input_tensor, model, original_image):
    # We target the last convolutional layer. 
    # Adjust 'features' or the specific layer name based on your model.py architecture
    # Usually it's model.features[-1] or the last conv layer in your forward pass.
    
    # 1. Hook into the last conv layer (Assume it's named 'conv_layers' or similar)
    # If your model has a 'features' block, use model.features
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module # This gets the very last conv layer found
            
    activations = []
    def save_activation(module, input, output):
        activations.append(output)
    
    gradients = []
    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    handle_act = target_layer.register_forward_hook(save_activation)
    handle_grad = target_layer.register_full_backward_hook(save_gradient)

    # Forward pass
    model.zero_grad()
    output = model(input_tensor)
    idx = output.argmax(dim=1).item()
    
    # Backward pass for the winning class
    output[0, idx].backward()

    # Get weights from gradients
    grads = gradients[0]
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    
    # Create weighted combination of activations
    cam = torch.sum(weights * activations[0], dim=1, keepdim=True)
    cam = F.relu(cam) # Remove negative influence
    
    # Normalize
    cam -= cam.min()
    cam /= cam.max()
    cam = cam.detach().cpu().numpy()[0, 0]

    # Resize and Overlay
    heatmap = cv2.resize(cam, (128, 128))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert original PIL to OpenCV format
    orig_cv = cv2.cvtColor(np.array(original_image.resize((128, 128))), cv2.COLOR_RGB2BGR)
    
    # Superimpose
    result = cv2.addWeighted(orig_cv, 0.6, heatmap_color, 0.4, 0)
    
    # Cleanup hooks
    handle_act.remove()
    handle_grad.remove()

    # Encode to Base64
    _, buffer = cv2.imencode('.jpg', result)
    return base64.b64encode(buffer).decode('utf-8')

# --- HELPER FUNCTIONS ---
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_for_heatmap = img.copy()
    img = img.resize((128, 128))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    tensor = torch.from_numpy(img_array).unsqueeze(0).to(device)
    return tensor, original_for_heatmap

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
        input_tensor, original_img = preprocess_image(content)
        input_tensor.requires_grad = True # Needed for Grad-CAM

        # 1. Standard Prediction
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            conf, pred_idx = torch.max(probabilities, 1)

        # 2. Heatmap Generation (Requires gradients)
        model.train() # Set to train briefly to allow gradient flow for Grad-CAM
        heatmap_base64 = generate_heatmap(input_tensor, model,original_img)
        model.eval()

        return {
            "tumor_type": CLASS_NAMES[pred_idx.item()],
            "confidence": round(conf.item() * 100, 2),
            "heatmap": heatmap_base64, # The magic "X-Factor"
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