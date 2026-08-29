"""
synthesize_audio.py

Converts text prompts (from generate_prompts.py) into audio files using
edge-tts: a free, open wrapper around Microsoft Edge's neural TTS voices.
No API key, no account, no paid tier.

Why edge-tts instead of Coqui/XTTS: XTTS is a heavy model whose dependencies
(older transformers internals) conflict with the modern transformers version
we need for Whisper fine-tuning in the same environment -- installing both
in one Colab runtime breaks one or the other. edge-tts has no ML framework
dependency at all (it just calls Microsoft's TTS service over the network),
so it can't conflict with anything else we install for training.

Bonus: edge-tts ships many high-quality neural voices across accents
out of the box (US / UK / Indian / Australian English) which gives us
genuine voice + accent diversity for free -- arguably more useful for
robustness than XTTS's default speaker set, and directly relevant if the
target usage involves Indian-English speakers.

Meant to run on Colab (T4 or even CPU-only, no GPU needed for this step).
Install first:
    pip install edge-tts soundfile
    (also needs ffmpeg on PATH to convert mp3 -> wav; Colab has it preinstalled)

Run:
    python synthesize_audio.py --prompts prompts.json --output_dir audio_raw/
"""

import argparse
import asyncio
import json
import os
import random
import subprocess
import tempfile

import soundfile as sf


# A mix of accents/genders for voice diversity. All free, built into edge-tts.
VOICES = [
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-GB-RyanNeural",
    "en-GB-SoniaNeural",
    "en-IN-NeerjaNeural",
    "en-IN-PrabhatNeural",
    "en-AU-NatashaNeural",
    "en-AU-WilliamNeural",
]

# Small rate/pitch jitter per sample so the same voice doesn't sound
# identically monotone across every prompt it's used for.
RATES = ["-10%", "-5%", "+0%", "+5%", "+10%"]
PITCHES = ["-5Hz", "+0Hz", "+5Hz"]


async def _synthesize_one(text: str, voice: str, rate: str, pitch: str, mp3_path: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(mp3_path)


def _mp3_to_wav(mp3_path: str, wav_path: str, sample_rate: int):
    # ffmpeg ships preinstalled on Colab; -y overwrites, -ac 1 forces mono
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", mp3_path,
            "-ar", str(sample_rate), "-ac", "1",
            wav_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def synthesize(prompts_path: str, output_dir: str, sample_rate: int = 16000, voices_per_prompt: int = 2):
    os.makedirs(output_dir, exist_ok=True)

    with open(prompts_path) as f:
        prompts = json.load(f)

    manifest = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for prompt in prompts:
            chosen_voices = random.sample(VOICES, min(voices_per_prompt, len(VOICES)))

            for voice in chosen_voices:
                rate = random.choice(RATES)
                pitch = random.choice(PITCHES)

                out_name = f"{prompt['id']}_{voice}"
                mp3_path = os.path.join(tmp_dir, f"{out_name}.mp3")
                wav_path = os.path.join(output_dir, f"{out_name}.wav")

                try:
                    asyncio.run(_synthesize_one(prompt["text"], voice, rate, pitch, mp3_path))
                    _mp3_to_wav(mp3_path, wav_path, sample_rate)
                except Exception as e:
                    print(f"  [skip] {prompt['id']} ({voice}): {e}")
                    continue
                finally:
                    if os.path.exists(mp3_path):
                        os.remove(mp3_path)

                manifest.append({
                    "audio_path": wav_path,
                    "text": prompt["text"],
                    "category": prompt["category"],
                    "voice": voice,
                    "rate": rate,
                    "pitch": pitch,
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
    parser.add_argument("--voices_per_prompt", type=int, default=2,
                         help="How many different voices to render each prompt in (for diversity).")
    args = parser.parse_args()

    synthesize(args.prompts, args.output_dir, args.sample_rate, args.voices_per_prompt)
