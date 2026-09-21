# Domain-Adaptable, Noise-Robust Speech-to-Text

A speech-to-text pipeline built by fine-tuning OpenAI's Whisper with
LoRA (parameter-efficient fine-tuning), designed to adapt to a target
vocabulary domain and stay robust under background noise — end-to-end,
from synthetic data generation through a served API and UI.

**Why this exists:** built while preparing to apply for a Data Science
Internship where one of the responsibilities is building a speech-to-text
feature for an Industry 4.0 SaaS product. Rather than just study ASR
theory, I wanted to build a real, working, benchmarked pipeline that
demonstrates the same underlying problem: adapting a general-purpose
speech model to a specific technical vocabulary, reliably, under noisy
real-world conditions — entirely with free/open tools.

> This project intentionally does **not** assume one narrow deployment
> scenario. Manufacturing / data-logging vocabulary is used as the
> primary example domain (numbers, measurements, technical terms) since
> it's directly relevant to Industry 4.0 / manufacturing analytics
> products, but the pipeline and code are domain-agnostic — swap the
> prompt list and it adapts to any vocabulary.

---

## Problem framing

Voice-based data entry / dictation in industrial or technical settings
has two properties that make off-the-shelf ASR underperform:

1. **Domain vocabulary** — numeric readings and technical terms
   (e.g. "moisture content", "compactability", "mold hardness") that
   general-purpose ASR models weren't specifically tuned for.
2. **Background noise** — real environments aren't studio-quiet.

So the goal isn't just "transcribe speech" — it's "transcribe *this*
kind of speech, reliably, under *these* conditions." That reframing
drives every design decision below.

## Architecture

```
Text prompts (domain + general vocabulary)
        │
        ▼
  TTS synthesis (edge-tts, multi-voice/multi-accent)
        │
        ▼
  Noise augmentation (ESC-50 machinery/industrial clips + audiomentations)
        │
        ▼
  Feature extraction (Whisper mel-spectrogram + tokenization)
        │
        ▼
  LoRA fine-tuning of whisper-small (PEFT)
        │
        ▼
  Evaluation: WER, baseline vs fine-tuned, clean vs noisy
        │
        ▼
  Inference wrapper → FastAPI (/transcribe) → Streamlit UI
```

**Why fine-tune instead of training from scratch?** Whisper was
pretrained on 680,000 hours of multilingual audio. Training a
comparable model from scratch needs thousands of GPU-hours — infeasible
on a free Colab T4. Fine-tuning adapts an already-strong model to a
specific domain in hours, which is also how ASR adaptation is actually
done in industry.

**Why LoRA specifically?** Full fine-tuning of whisper-small updates
~244M parameters — slow and memory-heavy on a T4. LoRA freezes the base
model and injects small trainable rank-decomposition matrices into the
attention projections, cutting trainable parameters by orders of
magnitude while retaining most of the adaptation benefit. Practical for
free-tier compute, and a technique worth knowing regardless of task.

**Why synthetic data?** No time/budget to record real domain audio at
scale. Instead: generate realistic domain-specific text prompts →
synthesize with edge-tts (free Microsoft neural voices spanning US/UK/
Indian/Australian English accents, for genuine voice diversity) →
overlay real machinery/industrial background noise. This is a
legitimate, industry-used technique for bootstrapping ASR training data
when labeled domain audio doesn't exist yet — not a shortcut hack.

**Why edge-tts specifically?** It's free, needs no API key, and — unlike
heavier local TTS models (e.g. Coqui/XTTS) — has no deep learning
framework dependency of its own, so it can't conflict with the
`transformers` version needed for Whisper fine-tuning in the same
environment.

## Repo structure

```
data/
  generate_prompts.py    # domain + general text prompt generation
  synthesize_audio.py    # TTS synthesis (edge-tts, multi-voice/multi-accent)
  augment_noise.py       # ESC-50 noise overlay via audiomentations
training/
  prepare_dataset.py     # HF Dataset build: features + tokenization + splits
  finetune_lora.py       # LoRA fine-tuning of whisper-small
  run_evaluation.py            # WER: baseline vs fine-tuned, clean vs noisy
inference/
  model_wrapper.py       # loads base model + LoRA adapter, transcribe()
  api.py                 # FastAPI /transcribe endpoint
app/
  streamlit_app.py       # UI on top of the API (upload or record audio)
notebooks/
  colab_training.ipynb   # full pipeline, runnable end-to-end on a free T4
results/
  wer_comparison.md      # benchmark results (filled in after training)
```

## Running it

Everything runs free, on Google Colab's free T4 GPU tier — no paid
APIs, no local GPU required.

1. Open `notebooks/colab_training.ipynb` in Colab, set runtime to T4 GPU
2. Run cells top to bottom — mounts Drive for checkpoint persistence,
   installs dependencies, generates data, trains, evaluates
3. Locally (or in another Colab cell), serve the model:
   ```bash
   uvicorn inference.api:app --host 0.0.0.0 --port 8000
   streamlit run app/streamlit_app.py
   ```

## Results

See [`results/wer_comparison.md`](results/wer_comparison.md) for Word
Error Rate (WER) comparisons: baseline `whisper-small` vs LoRA
fine-tuned, on clean vs noise-augmented test audio.
Word Error Rate (WER) — lower is better. Reported separately for clean
and noise-augmented test audio to isolate the noise-robustness gain from
the domain-adaptation gain.
 
| Condition | Baseline (whisper-small) | Fine-tuned (+ LoRA) | Relative improvement |
|---|---|---|---|
| Clean     | 43.66% | 1.52%  | 96.5% |
| Noisy     | 42.32% | 12.36% | 70.8% |
| Overall   | 43.26% | 4.78%  | 89.0% |

## What I'd do with more time/compute

- Real recorded domain audio (not just TTS-synthetic) for a production-
  grade dataset
- Larger base model (`whisper-medium`) for a stronger ceiling
- Streaming/real-time transcription instead of batch file upload
- Broader noise bank beyond ESC-50's machinery category
