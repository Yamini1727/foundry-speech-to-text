"""
run_evaluation.py

Computes Word Error Rate (WER) — the standard ASR benchmark metric — broken
down by:
  - model: baseline (vanilla pretrained whisper-small) vs fine-tuned (+ LoRA)
  - condition: clean vs noisy audio

This produces the "before vs after" evidence for the README, which is the
whole point of doing this as a benchmarked project rather than "I fine-tuned
something and it seems to work."

Run:
    python evaluate.py --dataset_dir hf_dataset/ --base_model openai/whisper-small \
        --lora_adapter whisper-lora-foundry/ --output results/wer_comparison.json
"""

import argparse
import json

import evaluate as hf_evaluate
import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def transcribe_batch(model, processor, dataset, device, batch_size=8):
    model.eval()
    predictions, references = [], []

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        input_features = torch.tensor(batch["input_features"]).to(device)

        with torch.no_grad():
            generated_ids = model.generate(input_features, max_new_tokens=128)

        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

        labels = [[tok if tok != -100 else processor.tokenizer.pad_token_id for tok in seq]
                  for seq in batch["labels"]]
        refs = processor.batch_decode(labels, skip_special_tokens=True)

        predictions.extend(preds)
        references.extend(refs)

    return predictions, references


def compute_wer_by_condition(predictions, references, conditions, wer_metric):
    results = {}
    for cond in set(conditions):
        idx = [i for i, c in enumerate(conditions) if c == cond]
        preds = [predictions[i] for i in idx]
        refs = [references[i] for i in idx]
        results[cond] = round(100 * wer_metric.compute(predictions=preds, references=refs), 2)

    results["overall"] = round(100 * wer_metric.compute(predictions=predictions, references=references), 2)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="hf_dataset")
    parser.add_argument("--base_model", default="openai/whisper-small")
    parser.add_argument("--lora_adapter", default="whisper-lora-foundry")
    parser.add_argument("--output", default="results/wer_comparison.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_from_disk(args.dataset_dir)
    test_set = dataset["test"]
    conditions = test_set["condition"]

    processor = WhisperProcessor.from_pretrained(args.base_model, language="English", task="transcribe")
    wer_metric = hf_evaluate.load("wer")

    print("=== Evaluating BASELINE (vanilla pretrained Whisper) ===")
    baseline_model = WhisperForConditionalGeneration.from_pretrained(args.base_model).to(device)
    baseline_preds, refs = transcribe_batch(baseline_model, processor, test_set, device)
    baseline_wer = compute_wer_by_condition(baseline_preds, refs, conditions, wer_metric)
    print(json.dumps(baseline_wer, indent=2))
    del baseline_model
    torch.cuda.empty_cache()

    print("\n=== Evaluating FINE-TUNED (base + LoRA adapter) ===")
    ft_model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    ft_model = PeftModel.from_pretrained(ft_model, args.lora_adapter).to(device)
    ft_preds, refs = transcribe_batch(ft_model, processor, test_set, device)
    ft_wer = compute_wer_by_condition(ft_preds, refs, conditions, wer_metric)
    print(json.dumps(ft_wer, indent=2))

    final_results = {
        "baseline_whisper_small": baseline_wer,
        "finetuned_lora": ft_wer,
        "relative_improvement_percent": {
            cond: round(100 * (baseline_wer[cond] - ft_wer[cond]) / baseline_wer[cond], 1)
            if baseline_wer[cond] > 0 else None
            for cond in baseline_wer
        },
        "sample_predictions": [
            {"reference": r, "baseline_pred": bp, "finetuned_pred": fp}
            for r, bp, fp in list(zip(refs, baseline_preds, ft_preds))[:10]
        ],
    }

    with open(args.output, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nSaved comparison -> {args.output}")
    print("\n--- SUMMARY (WER %, lower is better) ---")
    print(f"{'Condition':<10} {'Baseline':>10} {'Fine-tuned':>12} {'Improvement':>12}")
    for cond in baseline_wer:
        imp = final_results["relative_improvement_percent"][cond]
        print(f"{cond:<10} {baseline_wer[cond]:>9.2f}% {ft_wer[cond]:>11.2f}% {imp:>11.1f}%")


if __name__ == "__main__":
    main()
