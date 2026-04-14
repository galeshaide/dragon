from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import os
import numpy as np
import cv2
import uvicorn
import json

import tensorflow as tf

app = FastAPI()

# =====================================
# Enable CORS
# =====================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# Load Model
# =====================================
MODEL_PATH = "skin_cancer_model.keras"
CLASS_PATH = "skin_classes.json"

model = None
classes = None

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Skin Cancer Model Loaded Successfully!")
    else:
        print("❌ Model file not found")

    if os.path.exists(CLASS_PATH):
        with open(CLASS_PATH, "r") as f:
            class_indices = json.load(f)
        classes = {v: k for k, v in class_indices.items()}
        print("✅ Class labels loaded")
    else:
        print("⚠️ class file missing")
except Exception as e:
    print("❌ Error loading model:", e)


# =====================================
# Image Preprocessing
# =====================================
def preprocess_image(image_bytes):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img


# =====================================
# Prediction
# =====================================
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    try:
        input_image = preprocess_image(image_bytes)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    if model is None:
        return {"prediction": "Model not loaded"}

    try:
        prediction = model.predict(input_image)
        predicted_index = int(np.argmax(prediction))

        # Use real class names if available
        if classes:
            predicted_class = classes[predicted_index]
        else:
            predicted_class = f"class_{predicted_index}"

        return {
            "model": "skin",
            "prediction": predicted_class
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================
# Run Server
# =====================================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8012)