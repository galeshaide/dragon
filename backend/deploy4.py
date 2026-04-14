#brain tumor
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
import os
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Try to import PyTorch (optional) — if not available we'll return mock predictions
TRY_TORCH = True
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.models as models
except Exception:
    torch = None
    TRY_TORCH = False

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Class labels
CLASS_TYPES = ['glioma', 'meningioma', 'notumor', 'pituitary']
N_CLASSES = len(CLASS_TYPES)

# Device / model loading — support missing torch or missing model file by using a mock
model = None
device = None
MODEL_FILE = os.path.join(os.path.dirname(__file__), "brain_tumor_classifier.pth")
if TRY_TORCH and os.path.exists(MODEL_FILE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using Device: {device}")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model = model.to(device)
    model.eval()
    print("✅ Brain Tumor Classifier Model Loaded Successfully!")
else:
    if not TRY_TORCH:
        print("⚠️ PyTorch not available — deploy4 will return mock predictions.")
    else:
        print(f"⚠️ Model file not found at {MODEL_FILE} — deploy4 will return mock predictions.")

# ✅ Image Preprocessing (only when torchvision available)
if TRY_TORCH:
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
else:
    transform = None

# ✅ Prediction Endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Image preprocessing failed: {e}"})

    # If model not loaded, return a sensible mock prediction
    if model is None:
        return JSONResponse(status_code=200, content={"prediction": "notumor", "note": "mock prediction — model missing or torch unavailable"})

    try:
        # Preprocess image
        image = transform(image).unsqueeze(0).to(device)

        # Get Prediction
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            tumor_type = CLASS_TYPES[predicted.item()]

        return JSONResponse(status_code=200, content={"prediction": tumor_type})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Model inference failed: {str(e)}"})

# ✅ Run the FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004)
