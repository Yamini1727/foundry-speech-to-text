"""
streamlit_app.py

Lightweight UI on top of the FastAPI backend — lets anyone (including a
non-technical reviewer) upload/record audio and see the transcription,
without needing to hit the API with curl/Postman.

Run:
    streamlit run streamlit_app.py

Set API_URL env var if the FastAPI backend isn't on localhost:8000
(e.g. if deployed separately on a free host).
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Speech-to-Text Demo", page_icon="🎙️")

st.title("🎙️ Domain-Adaptable Speech-to-Text")
st.caption(
    "Fine-tuned Whisper (LoRA) — noise-robust ASR adapted for technical / "
    "manufacturing data-logging vocabulary as an example domain."
)

with st.sidebar:
    st.subheader("About this demo")
    st.write(
        "- Base model: `openai/whisper-small`\n"
        "- Fine-tuned with LoRA (PEFT) on synthetic domain audio\n"
        "- Trained with noise augmentation for robustness\n"
        "- See `results/wer_comparison.md` in the repo for benchmark numbers"
    )
    st.text_input("API URL", value=API_URL, key="api_url", disabled=True)

tab1, tab2 = st.tabs(["Upload audio file", "Record from microphone"])

audio_bytes = None
filename = "audio.wav"

with tab1:
    uploaded_file = st.file_uploader("Upload a .wav / .mp3 / .flac file", type=["wav", "mp3", "flac", "m4a", "ogg"])
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        filename = uploaded_file.name
        st.audio(audio_bytes)

with tab2:
    recorded = st.audio_input("Record a short clip")
    if recorded is not None:
        audio_bytes = recorded.read()
        filename = "recording.wav"
        st.audio(audio_bytes)

if audio_bytes is not None:
    if st.button("Transcribe", type="primary"):
        with st.spinner("Transcribing..."):
            try:
                response = requests.post(
                    f"{st.session_state.api_url}/transcribe",
                    files={"file": (filename, audio_bytes)},
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()

                st.success("Transcription complete")
                st.text_area("Transcribed text", value=result["text"], height=120)
                st.caption(f"Model used: {result['model_type']}")

            except requests.exceptions.ConnectionError:
                st.error(
                    f"Could not reach the API at {st.session_state.api_url}. "
                    "Make sure the FastAPI backend (inference/api.py) is running."
                )
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Upload a file or record audio above to get started.")
