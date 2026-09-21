"""
finetune_lora.py

Parameter-efficient fine-tuning of Whisper-small using LoRA (via PEFT).
Chosen specifically because we're training on a free Colab T4 in a short
timeframe — full fine-tuning of even whisper-small on limited GPU memory /
time is unnecessarily expensive when LoRA gets us most of the adaptation
benefit at a fraction of the trainable parameters and training time.

Run (on Colab):
    python finetune_lora.py --dataset_dir hf_dataset/ --output_dir whisper-lora-foundry/ \
        --model_name openai/whisper-small --epochs 4 --batch_size 8
"""

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="hf_dataset")
    parser.add_argument("--model_name", default="openai/whisper-small")
    parser.add_argument("--output_dir", default="whisper-lora-foundry")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = load_from_disk(args.dataset_dir)

    print(f"Loading base model: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(args.model_name, language="English", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    print("Applying LoRA adapters to attention projections (q_proj, v_proj)...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # sanity check: should be << full model params

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=50,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=128,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to=["none"],
        remove_unused_columns=False,  # required for PEFT + custom collator
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    print("Starting LoRA fine-tuning...")
    trainer.train()

    print(f"Saving LoRA adapter -> {args.output_dir}")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print("Done. Note: this saves ONLY the small LoRA adapter weights, "
          "not a full copy of Whisper — keeps the artifact lightweight for GitHub.")


if __name__ == "__main__":
    main()
