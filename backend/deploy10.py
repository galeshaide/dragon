from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import pdfplumber
from gtts import gTTS
import os
import io
import uvicorn

app = FastAPI()

# ==========================================
# ✅ Enable CORS (Frontend Access)
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ✅ Load Summarization Model (Optional)
# ==========================================
summarizer = None
try:
    from transformers import pipeline
    summarizer = pipeline("summarization")
    print("✅ Summarization model loaded successfully.")
except Exception as e:
    summarizer = None
    print("⚠ Transformers not available. Using fallback summary.")
    print("Error:", e)

# ==========================================
# ✅ Ensure Static Directory Exists
# ==========================================
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# ==========================================
# ✅ Extract Text From PDF
# ==========================================
def extract_pdf_text(pdf_bytes):
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = "\n".join(
                page.extract_text() for page in pdf.pages if page.extract_text()
            )
        return text if text.strip() != "" else None
    except Exception:
        return None


# ==========================================
# ✅ Prediction Endpoint
# ==========================================
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only PDF files are supported."}
        )

    pdf_bytes = await file.read()

    # Extract text
    text = extract_pdf_text(pdf_bytes)
    if text is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid PDF or no readable text found."}
        )

    # ----------------------------------
    # 🔹 Summarize Text
    # ----------------------------------
    if summarizer:
        try:
            summary_output = summarizer(
                text[:2000],  # Limit long input
                max_length=120,
                min_length=40,
                do_sample=False
            )
            summary = summary_output[0]["summary_text"]
        except Exception:
            summary = text[:500]  # fallback if summarization fails
    else:
        summary = text[:500]  # fallback summary

    # ----------------------------------
    # 🔹 Convert Summary to Speech
    # ----------------------------------
    audio_filename = "summary_audio.mp3"
    audio_path = os.path.join(STATIC_DIR, audio_filename)

    try:
        tts = gTTS(text=summary, lang="en")
        tts.save(audio_path)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Text-to-speech failed: {str(e)}"}
        )

    # ----------------------------------
    # 🔹 Return Result
    # ----------------------------------
    return {
        "summary": summary,
        "audio_url": f"http://127.0.0.1:8010/static/{audio_filename}"
    }


# ==========================================
# ✅ Serve Audio File Properly
# ==========================================
@app.get("/static/{filename}")
async def serve_static(filename: str):
    file_path = os.path.join(STATIC_DIR, filename)

    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            media_type="audio/mpeg",
            filename=filename
        )

    return JSONResponse(
        status_code=404,
        content={"error": "Audio file not found."}
    )


# ==========================================
# ✅ Run Server
# ==========================================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)