"""
augment_noise.py

Takes clean synthesized audio (from synthesize_audio.py) and creates
noise-augmented copies at varying SNR levels, using free noise sources:

  - ESC-50 dataset (free, CC-licensed, has a "machinery/industrial" category
    among its 50 environmental sound classes) for background noise clips.
  - audiomentations library for mixing / SNR control / reverb / pitch shift.

We keep a portion of clean (unaugmented) audio too, so the final dataset is
a MIX of clean + noisy — this lets us report WER on both splits separately
(see training/evaluate.py) and demonstrate the noise-robustness gain
explicitly, rather than only training/testing on noisy audio.

Install (Colab):
    pip install audiomentations soundfile
    # ESC-50 download handled below via git clone (public GitHub repo)

Run:
    python augment_noise.py --manifest audio_raw/manifest.json --output_dir audio_augmented/
"""

import argparse
import glob
import json
import os
import random
import subprocess

import soundfile as sf
from audiomentations import Compose, AddBackgroundNoise, Gain, PitchShift


ESC50_REPO = "https://github.com/karolpiczak/ESC-50.git"
ESC50_MACHINERY_KEYWORDS = ["engine", "machine", "vacuum", "drill", "chainsaw", "washing_machine"]


def ensure_esc50(local_dir: str = "ESC-50"):
    if not os.path.exists(local_dir):
        print("Cloning ESC-50 (free, CC-BY licensed environmental sound dataset)...")
        subprocess.run(["git", "clone", "--depth", "1", ESC50_REPO, local_dir], check=True)
    audio_dir = os.path.join(local_dir, "audio")
    meta_csv = os.path.join(local_dir, "meta", "esc50.csv")
    return audio_dir, meta_csv


def get_machinery_noise_files(audio_dir: str, meta_csv: str):
    import csv
    files = []
    with open(meta_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["category"].lower()
            if any(k in category for k in ESC50_MACHINERY_KEYWORDS):
                files.append(os.path.join(audio_dir, row["filename"]))
    if not files:
        # fallback: just use all ESC-50 clips as generic ambient noise
        files = glob.glob(os.path.join(audio_dir, "*.wav"))
    return files


def build_augmenter(noise_dir_files):
    return Compose([
        AddBackgroundNoise(sounds_path=noise_dir_files, min_snr_db=3.0, max_snr_db=15.0, p=1.0),
        Gain(min_gain_db=-3, max_gain_db=3, p=0.3),
        PitchShift(min_semitones=-1, max_semitones=1, p=0.2),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="audio_raw/manifest.json")
    parser.add_argument("--output_dir", default="audio_augmented")
    parser.add_argument("--noise_fraction", type=float, default=0.5,
                         help="Fraction of samples to create a noisy copy for (rest stay clean-only in final dataset)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    esc50_audio_dir, esc50_meta = ensure_esc50()
    noise_files = get_machinery_noise_files(esc50_audio_dir, esc50_meta)
    print(f"Found {len(noise_files)} machinery/industrial-ish noise clips")

    augmenter = build_augmenter(noise_files)

    with open(args.manifest) as f:
        clean_manifest = json.load(f)

    final_manifest = []

    for entry in clean_manifest:
        # always keep the clean version
        final_manifest.append({**entry, "condition": "clean"})

        if random.random() < args.noise_fraction:
            audio, sr = sf.read(entry["audio_path"])
            noisy = augmenter(samples=audio.astype("float32"), sample_rate=sr)

            noisy_path = os.path.join(
                args.output_dir,
                os.path.basename(entry["audio_path"]).replace(".wav", "_noisy.wav"),
            )
            sf.write(noisy_path, noisy, sr)

            final_manifest.append({
                **entry,
                "audio_path": noisy_path,
                "condition": "noisy",
            })

    out_manifest_path = os.path.join(args.output_dir, "manifest_full.json")
    with open(out_manifest_path, "w") as f:
        json.dump(final_manifest, f, indent=2)

    n_clean = sum(1 for e in final_manifest if e["condition"] == "clean")
    n_noisy = sum(1 for e in final_manifest if e["condition"] == "noisy")
    print(f"Final dataset: {n_clean} clean + {n_noisy} noisy = {len(final_manifest)} total")
    print(f"Manifest -> {out_manifest_path}")


if __name__ == "__main__":
    main()
