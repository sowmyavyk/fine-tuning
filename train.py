"""Standalone LoRA fine-tuning for FinLens.

Reads the saved train/eval split (data/fintech_train.json, data/fintech_eval.json),
fine-tunes Qwen2.5-1.5B-4bit, and saves the LoRA adapter to
fintech_finetuned_qwen/adapters.

Run safely in the background (Mac stays awake during training):
    caffeinate -d -i -s nohup .venv/bin/python3.11 -u train.py > /tmp/train.log 2>&1 &

Monitor:
    tail -f /tmp/train.log
"""

import json
import os
from pathlib import Path

from fintech_data import FintechFineTuner, DataProcessor

BASE_MODEL = "mlx-community/Qwen2.5-1.5B-4bit"
OUTPUT_DIR = "fintech_finetuned_qwen"

DEFAULTS = {
    "num_epochs": int(os.getenv("EPOCHS", "2")),
    "batch_size": int(os.getenv("BATCH", "2")),
    "learning_rate": float(os.getenv("LR", "2e-4")),
    "gradient_accumulation_steps": int(os.getenv("GRAD_ACCUM", "4")),
    "max_seq_length": int(os.getenv("MAX_SEQ", "1024")),
    "weight_decay": float(os.getenv("WEIGHT_DECAY", "0.01")),
    "warmup_ratio": float(os.getenv("WARMUP", "0.1")),
    "lr_scheduler_type": "cosine",
    "logging_steps": int(os.getenv("LOG_STEPS", "10")),
    "save_steps": int(os.getenv("SAVE_STEPS", "400")),
    "eval_steps": int(os.getenv("EVAL_STEPS", "1000")),
    "val_batches": int(os.getenv("VAL_BATCHES", "32")),
}


def main():
    processor = DataProcessor()
    prefix = "fintech_subset" if os.getenv("SUBSET", "0") == "1" else "fintech"
    train_data, eval_data = processor.load_dataset(prefix=prefix)
    print(f"Loaded: {len(train_data)} train, {len(eval_data)} eval (prefix={prefix})")

    tuner = FintechFineTuner(
        base_model_name=BASE_MODEL,
        output_dir=OUTPUT_DIR,
        max_seq_length=DEFAULTS["max_seq_length"],
    )
    tuner.load_base_model()
    tuner.apply_lora(r=16, lora_alpha=32)

    tuner.train(
        train_data=train_data,
        eval_data=eval_data,
        num_epochs=DEFAULTS["num_epochs"],
        batch_size=DEFAULTS["batch_size"],
        learning_rate=DEFAULTS["learning_rate"],
        gradient_accumulation_steps=DEFAULTS["gradient_accumulation_steps"],
        max_seq_length=DEFAULTS["max_seq_length"],
        weight_decay=DEFAULTS["weight_decay"],
        warmup_ratio=DEFAULTS["warmup_ratio"],
        lr_scheduler_type=DEFAULTS["lr_scheduler_type"],
        logging_steps=DEFAULTS["logging_steps"],
        save_steps=DEFAULTS["save_steps"],
        eval_steps=DEFAULTS["eval_steps"],
        val_batches=DEFAULTS["val_batches"],
    )
    print("Training complete. Adapter saved to fintech_finetuned_qwen/adapters")


if __name__ == "__main__":
    main()
