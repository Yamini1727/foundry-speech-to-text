"""
prepare_dataset.py

Converts our audio manifest (clean + noisy) into a HuggingFace `datasets`
object with Whisper input features + tokenized labels, then splits into
train / validation / test. Test split is further separated by `condition`
(clean vs noisy) so we can report WER on each separately later.

Run:
    python prepare_dataset.py --manifest audio_augmented/manifest_full.json \
        --output_dir hf_dataset/ --model_name openai/whisper-small
"""

import argparse
import json
import os

import numpy as np
from datasets import Dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer


def load_manifest_as_dataset(manifest_path: str) -> Dataset:
    with open(manifest_path) as f:
        records = json.load(f)

    ds = Dataset.from_list([
        {"audio": r["audio_path"], "text": r["text"], "condition": r.get("condition", "clean"),
         "category": r.get("category", "unknown")}
        for r in records
    ])
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


def prepare_split(ds: Dataset, feature_extractor, tokenizer):
    def _map_fn(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = tokenizer(batch["text"]).input_ids
        return batch

    return ds.map(_map_fn, remove_columns=["audio"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="audio_augmented/manifest_full.json")
    parser.add_argument("--output_dir", default="hf_dataset")
    parser.add_argument("--model_name", default="openai/whisper-small")
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading manifest...")
    ds = load_manifest_as_dataset(args.manifest)

    print("Splitting train / val / test (stratified-ish by condition via shuffle+split)...")
    split1 = ds.train_test_split(test_size=args.test_size, seed=args.seed)
    train_val, test = split1["train"], split1["test"]

    split2 = train_val.train_test_split(test_size=args.val_size, seed=args.seed)
    train, val = split2["train"], split2["test"]

    print(f"train={len(train)}  val={len(val)}  test={len(test)}")
    print("Test split condition breakdown:",
          {c: sum(1 for x in test["condition"] if x == c) for c in set(test["condition"])})

    print(f"Loading feature extractor + tokenizer for {args.model_name}...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language="English", task="transcribe")

    print("Extracting features (this does the heavy lifting: mel-spectrograms + tokenization)...")
    dataset_dict = DatasetDict({
        "train": prepare_split(train, feature_extractor, tokenizer),
        "validation": prepare_split(val, feature_extractor, tokenizer),
        "test": prepare_split(test, feature_extractor, tokenizer),
    })

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_dict.save_to_disk(args.output_dir)
    print(f"Saved processed dataset -> {args.output_dir}")


if __name__ == "__main__":
    main()
