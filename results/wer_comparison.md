# WER Benchmark Results

Word Error Rate (WER) — lower is better. Reported separately for clean
and noise-augmented test audio to isolate the noise-robustness gain from
the domain-adaptation gain.

| Condition | Baseline (whisper-small) | Fine-tuned (+ LoRA) | Relative improvement |
|---|---|---|---|
| Clean     | 43.66% | 1.52%  | 96.5% |
| Noisy     | 42.32% | 12.36% | 70.8% |
| Overall   | 43.26% | 4.78%  | 89.0% |

## Sample predictions

| Reference | Baseline prediction | Fine-tuned prediction |
|---|---|---|
| the server logs show no errors this morning | The server logs show no errors this morning. | the server logs show no errors this morning |
| green compactability at station one is 19 point 6 percent | Green Compactability at Station 1 is 19.6%. | green compactability at station one is 19 point 6 percent |
| sand temperature at station two is 37 degrees celsius | Sand temperature at station 2 is 37 degrees Celsius. | sand temperatur at station two is 37 degrees celsius |
| recording loss on ignition as 19 point 4 percent on station two | Recording loss on ignition as 19.4% on station 2. | recording loss on ignition as 19 point 4 percent on station two |
| dead clay content reading is 11 point 2 percent | Dead clay content reading is 11.2%. | deid clay content reading is 11 point 2 percent |

## Notes on interpreting these numbers

- **The baseline WER looks high (~43%) partly because of a text-formatting
  mismatch, not pure transcription error.** Whisper's baseline output uses
  natural punctuation, capitalization, and digit-formatted numbers
  (`"19.6%."`), while the reference/ground-truth text is written in the
  spelled-out, unpunctuated style the synthetic prompt generator produces
  (`"19 point 6 percent"`). A raw word-level WER penalizes this style
  mismatch heavily, inflating baseline WER for reasons unrelated to
  whether the audio was actually understood correctly.
- **The fine-tuned model's very low WER is driven in large part by it
  learning to reproduce that exact reference formatting** (spelled-out
  numbers, no punctuation, lowercase) — a legitimate part of domain
  adaptation, but it means part of the "improvement" reflects style
  matching rather than pure recognition accuracy. A fairer before/after
  comparison would normalize both predictions and references (lowercase,
  strip punctuation, standardize number formatting) before computing WER.
- The **noise-robustness signal is still meaningful**: fine-tuned WER on
  noisy audio (12.36%) is clearly higher than on clean audio (1.52%),
  showing the model isn't simply memorizing a fixed answer set — but with
  a small, templated synthetic dataset, some of the near-zero clean-split
  WER may still reflect the limited sentence-template pool rather than
  fully general noise robustness.
- Test set here is TTS-synthetic audio, not real human speech — a
  reasonable proof-of-concept signal, but real recorded audio would be
  needed to validate before any real deployment.
