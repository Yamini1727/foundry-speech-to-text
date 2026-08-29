"""
model_wrapper.py

Loads the base Whisper model + our fine-tuned LoRA adapter once, and exposes
a simple transcribe() function. Kept separate from api.py / streamlit_app.py
so both can import and share the same loaded model without duplicating
loading logic (avoids double GPU/CPU memory usage if both run in-process).
"""

import io
import os

import numpy as np
import soundfile as sf
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

DEFAULT_BASE_MODEL = os.environ.get("BASE_MODEL", "openai/whisper-small")
DEFAULT_ADAPTER_PATH = os.environ.get("LORA_ADAPTER_PATH", "whisper-lora-foundry")


class SpeechToTextModel:
    def __init__(self, base_model: str = DEFAULT_BASE_MODEL, adapter_path: str = DEFAULT_ADAPTER_PATH,
                 use_lora: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(base_model, language="English", task="transcribe")

        model = WhisperForConditionalGeneration.from_pretrained(base_model)

        self.using_lora = False
        if use_lora and os.path.isdir(adapter_path):
            try:
                model = PeftModel.from_pretrained(model, adapter_path)
                self.using_lora = True
            except Exception as e:
                print(f"[warn] Could not load LoRA adapter at '{adapter_path}': {e}. "
                      f"Falling back to base pretrained model.")

        self.model = model.to(self.device)
        self.model.eval()

    def transcribe(self, audio_bytes: bytes) -> str:
        """audio_bytes: raw bytes of a wav/flac/etc file (readable by soundfile)."""
        audio, sr = sf.read(io.BytesIO(audio_bytes))

        if audio.ndim > 1:  # stereo -> mono
            audio = audio.mean(axis=1)

        if sr != 16000:
            audio = self._resample(audio, sr, 16000)

        input_features = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(input_features, max_new_tokens=128)

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        import librosa
        return librosa.resample(audio.astype("float32"), orig_sr=orig_sr, target_sr=target_sr)


# Module-level singleton so api.py / streamlit_app.py share one loaded model
_model_instance = None


def get_model() -> SpeechToTextModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = SpeechToTextModel()
    return _model_instance
