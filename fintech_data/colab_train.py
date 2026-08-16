import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

TRAIN_PATH = "/content/finlens/fintech_train.json"
EVAL_PATH = "/content/finlens/fintech_eval.json"
OUT_DIR = "/content/drive/MyDrive/finlens_checkpoints"
FINAL_DIR = "/content/drive/MyDrive/finlens_adapter/final"
MODEL_NAME = "Qwen/Qwen2.5-1.5B"

# 1) Load YOUR data (uses the `text` field already in chat format)
with open(TRAIN_PATH) as f:
    train_data = json.load(f)
with open(EVAL_PATH) as f:
    eval_data = json.load(f)

train_ds = Dataset.from_list([{"text": x["text"]} for x in train_data])
eval_ds = Dataset.from_list([{"text": x["text"]} for x in eval_data])
print(f"Loaded {len(train_ds)} train, {len(eval_ds)} eval")

# 2) Load Qwen2.5-1.5B in fp16 (1.5B needs ~3GB — fits T4 easily, no 4-bit needed)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# 3) LoRA (r=16, alpha=32 — same as your local run)
model = get_peft_model(
    model,
    LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    ),
)
model.print_trainable_parameters()

# 4) Trainer — torch_compile=False avoids the Dynamo/fused-loss bug entirely
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=858,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    bf16=False,
    fp16=True,
    optim="adamw_torch",
    torch_compile=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    max_seq_length=512,
    dataset_text_field="text",
    packing=False,
)

# 5) Resume if a checkpoint already exists
last = None
if os.path.isdir(OUT_DIR):
    ckpts = sorted(
        [p for p in os.listdir(OUT_DIR) if p.startswith("checkpoint-")],
        key=lambda p: int(p.split("-")[1]),
    )
    if ckpts:
        last = os.path.join(OUT_DIR, ckpts[-1])
        print(f"Resuming from {last}")

trainer.train(resume_from_checkpoint=last)

# 6) Save the LoRA adapter (not the whole model) to Drive so it survives sessions
model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
print(f"DONE — adapter saved to {FINAL_DIR}")
