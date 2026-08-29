# WER Benchmark Results

> **Status: template.** Run `notebooks/colab_training.ipynb` end-to-end on
> a free Colab T4 to generate `results/wer_comparison.json`, then fill in
> the numbers below (or write a tiny script to auto-render this table from
> that JSON).

Word Error Rate (WER) — lower is better. Reported separately for clean
and noise-augmented test audio to isolate the noise-robustness gain from
the domain-adaptation gain.

| Condition | Baseline (whisper-small) | Fine-tuned (+ LoRA) | Relative improvement |
|---|---|---|---|
| Clean     | TBD % | TBD % | TBD % |
| Noisy     | TBD % | TBD % | TBD % |
| Overall   | TBD % | TBD % | TBD % |

## Sample predictions

| Reference | Baseline prediction | Fine-tuned prediction |
|---|---|---|
| _fill in from `results/wer_comparison.json` → `sample_predictions`_ | | |

## Notes on interpreting these numbers

- The **domain-adaptation gain** shows up as improved WER on
  manufacturing-vocabulary phrases specifically (technical terms the
  base model wasn't tuned for).
- The **noise-robustness gain** shows up as a smaller WER gap between
  clean and noisy conditions after fine-tuning, versus the baseline.
- Test set here is TTS-synthetic audio, not real human speech — a
  reasonable proof-of-concept signal, but real recorded audio would be
  needed to validate before any real deployment.
