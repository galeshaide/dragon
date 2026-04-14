from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import io
import os
from PIL import Image
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Load Models
# ==========================

LUNG_MODEL_PATH = "pulmonary_nodule_model.keras"
LUNG_SIZE_MODEL_PATH = "nodule_size_predictor.h5"

lung_model = None
lung_size_model = None

if os.path.exists(LUNG_MODEL_PATH):
    lung_model = tf.keras.models.load_model(LUNG_MODEL_PATH)
    print("✅ Lung classification model loaded")
else:
    print("❌ Lung classification model NOT found")

if os.path.exists(LUNG_SIZE_MODEL_PATH):
    lung_size_model = tf.keras.models.load_model(LUNG_SIZE_MODEL_PATH)
    print("✅ Lung size model loaded")
else:
    print("❌ Lung size model NOT found")

# ==========================
# Preprocessing
# ==========================

IMG_SIZE = 224

def preprocess_classification(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_regression(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==========================
# ✅ UNIFIED ROUTE (IMPORTANT FIX)
# ==========================

@app.post("/predict/")
async def unified_predict(file: UploadFile = File(...)):
    if lung_model is None:
        return JSONResponse(status_code=400, content={"error": "Lung model not loaded"})

    image_bytes = await file.read()
    input_data = preprocess_classification(image_bytes)

    prediction = lung_model.predict(input_data)
    predicted_class = np.argmax(prediction)

    classes = ["Normal", "Cyst", "Stone", "Tumor"]

    return {
        "model": "lung",
        "prediction": classes[predicted_class]
    }

# ==========================
# Lung Classification (Original)
# ==========================

@app.post("/predict/lung/classification/")
async def predict_lung(file: UploadFile = File(...)):
    return await unified_predict(file)

# ==========================
# Lung Size Regression
# ==========================

@app.post("/predict/lung_size/regression/")
async def predict_lung_size(file: UploadFile = File(...)):
    if lung_size_model is None:
        return JSONResponse(status_code=400, content={"error": "Lung size model not loaded"})

    image_bytes = await file.read()
    input_data = preprocess_regression(image_bytes)

    prediction = lung_size_model.predict(input_data)
    size_value = float(prediction[0][0])

    return {
        "model": "lung_size",
        "predicted_size_mm": round(size_value, 2)
    }

# ==========================
# OPTIONAL: unified lung_size route (future use)
# ==========================

@app.post("/predict/")
async def unified_predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    if lung_model is None:
        return {
            "model": "lung",
            "prediction": "Tumor (Mock Prediction)",
            "note": "Model not available"
        }

    input_data = preprocess_classification(image_bytes)
    prediction = lung_model.predict(input_data)
    predicted_class = np.argmax(prediction)

    classes = ["Normal", "Cyst", "Stone", "Tumor"]

    return {
        "model": "lung",
        "prediction": classes[predicted_class]
    }

# ==========================
# Run Server
# ==========================

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)