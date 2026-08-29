"""
synthesize_audio.py

Converts text prompts (from generate_prompts.py) into audio files using
Coqui TTS (free, open source, runs locally/on Colab — no paid API keys).

We use multiple speaker voices / models where available to introduce voice
diversity, since a single monotone TTS voice would make the fine-tuned model
brittle to anything that isn't that exact voice.

Meant to run on Colab (T4). Install first:
    pip install TTS soundfile

Run:
    python synthesize_audio.py --prompts prompts.json --output_dir audio_raw/
"""

import argparse
import json
import os
import random

import soundfile as sf


# Multiple free Coqui TTS models to get voice diversity.
# (VCTK is a real multi-speaker model with 100+ speakers -> lots of diversity
# from a single model download.)
TTS_MODELS = [
    "tts_models/en/vctk/vits",          # multi-speaker (~109 speakers)
    "tts_models/en/ljspeech/tacotron2-DDC",  # single speaker, different timbre
]


def synthesize(prompts_path: str, output_dir: str, sample_rate: int = 16000, max_speakers_per_model: int = 8):
    from TTS.api import TTS

    os.makedirs(output_dir, exist_ok=True)

    with open(prompts_path) as f:
        prompts = json.load(f)

    manifest = []

    for model_name in TTS_MODELS:
        print(f"Loading TTS model: {model_name}")
        tts = TTS(model_name)

        speakers = None
        if getattr(tts, "speakers", None):
            speakers = random.sample(tts.speakers, min(max_speakers_per_model, len(tts.speakers)))

        for prompt in prompts:
            speaker = random.choice(speakers) if speakers else None
            out_name = f"{prompt['id']}_{model_name.split('/')[-1]}"
            if speaker:
                out_name += f"_{speaker}"
            out_path = os.path.join(output_dir, f"{out_name}.wav")

            try:
                if speaker:
                    tts.tts_to_file(text=prompt["text"], speaker=speaker, file_path=out_path)
                else:
                    tts.tts_to_file(text=prompt["text"], file_path=out_path)
            except Exception as e:
                print(f"  [skip] {prompt['id']} ({model_name}): {e}")
                continue

            manifest.append({
                "audio_path": out_path,
                "text": prompt["text"],
                "category": prompt["category"],
                "tts_model": model_name,
                "speaker": speaker,
            })

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Synthesized {len(manifest)} audio files -> {output_dir}")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="prompts.json")
    parser.add_argument("--output_dir", default="audio_raw")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_speakers_per_model", type=int, default=8)
    args = parser.parse_args()

    synthesize(args.prompts, args.output_dir, args.sample_rate, args.max_speakers_per_model)
