#t2 lung
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import numpy as np
import io
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import cv2

# TensorFlow is optional in this environment. Guard the import so the
# service can run with mock responses when TF is not installed.
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    tf = None
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available — deploy1 will return mock predictions.")

app = FastAPI()

# ✅ Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attempt to load model, otherwise fallback to mock predictions
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pulmonary_nodule_model.keras")
model = None
if TF_AVAILABLE:
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Pulmonary Nodule Classification Model Loaded Successfully!")
        else:
            print(f"⚠️ Pulmonary model not found at {MODEL_PATH} — deploy1 will return mock predictions.")
    except Exception as e:
        model = None
        print(f"⚠️ Failed to load pulmonary model ({e}) — deploy1 will return mock predictions.")
else:
    model = None

# ✅ Class labels
CLASS_LABELS = ["Benign", "Malignant", "Normal"]

# ✅ Preprocessing function
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (224, 224))  # Resize
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# ✅ Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Image preprocessing failed: {str(e)}"})

    # If no real model available, return a mock
    if model is None:
        return JSONResponse(status_code=200, content={"prediction": "Normal", "confidence": "100.00%", "note": "mock prediction - model not loaded"})

    # 🔍 Model prediction
    try:
        prediction = model.predict(processed_image)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Model inference failed: {str(e)}"})

    return JSONResponse(status_code=200, content={
        "prediction": CLASS_LABELS[predicted_class],
        "confidence": f"{confidence:.2f}%"
    })

# ✅ Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
