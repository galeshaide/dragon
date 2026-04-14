#nodule size t23
from fastapi import FastAPI, File, UploadFile
import os
import numpy as np
import tensorflow as tf
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# ✅ Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000"],  # Change "*" to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load the trained model (support missing/absolute paths)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "nodule_size_predictor.h5")
if not os.path.exists(MODEL_PATH):
    # fall back to the absolute path in the original file if present
    MODEL_PATH = "E:/task1/nodule_size_predictor.h5"

model = None
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Nodule Size Prediction Model Loaded Successfully!")
    else:
        print(f"⚠️ Nodule size model not found (looked for {MODEL_PATH}). Using mock responses.")
except Exception as e:
    model = None
    print(f"⚠️ Could not load nodule size model ({e}). Using mock responses.")

# ✅ Function to preprocess image
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
        original_size = img.size
        img = img.resize((256, 256), Image.LANCZOS)  # Resize with high-quality resampling
        print(f"Resized nodule image from {original_size} to {img.size}")
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 256, 256, 1)
        return img_array
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

# ✅ Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        image_bytes = await file.read()
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        if processed_image is None:
            return {"error": "Image preprocessing failed"}

        # If model missing, return a mock prediction
        if model is None:
            return {"prediction": 0.0, "note": "mock prediction - model not loaded"}

        # Predict
        prediction = model.predict(processed_image)
        predicted_value = float(prediction[0][0])  # Convert to Python float

        return {"prediction": predicted_value}  # ✅ Only return prediction
    except Exception as e:
        return {"error": f"❌ Error: {str(e)}"}

# ✅ Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)
