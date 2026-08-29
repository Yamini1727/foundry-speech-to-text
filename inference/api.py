"""
api.py

Minimal FastAPI backend exposing a /transcribe endpoint. Kept intentionally
small/focused (single responsibility) rather than a kitchen-sink app —
this is the "production/backend" piece of the demo.

Run locally / on Colab (with ngrok or similar for a public URL) / on a free
host like Hugging Face Spaces (Docker SDK) or Render free tier:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference.model_wrapper import get_model

app = FastAPI(
    title="Domain-Adaptable Speech-to-Text API",
    description=(
        "Fine-tuned Whisper (LoRA) for noise-robust, domain-adapted speech "
        "recognition. Example domain: manufacturing / data-logging vocabulary."
    ),
    version="1.0.0",
)

# Permissive CORS since this is a demo project, not a production deployment
# with real user data.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscriptionResponse(BaseModel):
    text: str
    model_type: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)):
    allowed_ext = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
    if not file.filename.lower().endswith(allowed_ext):
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Use one of: {allowed_ext}")

    audio_bytes = await file.read()
    model = get_model()

    try:
        text = model.transcribe(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    return TranscriptionResponse(
        text=text,
        model_type="whisper-small + LoRA" if model.using_lora else "whisper-small (base, adapter not found)",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
