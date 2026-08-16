import json
from pathlib import Path
from typing import Optional

from datasets import Dataset

from mlx_tune import FastLanguageModel, SFTConfig, SFTTrainer


class FintechFineTuner:
    def __init__(
        self,
        base_model_name: str = "mlx-community/Qwen2.5-1.5B-4bit",
        output_dir: str = "finetuned_model",
        max_seq_length: int = 1024,
    ):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None
        self.adapter_path = None

    def load_base_model(self):
        print(f"Loading base model: {self.base_model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            self.base_model_name,
            max_seq_length=self.max_seq_length,
        )
        print("Base model loaded")

    def apply_lora(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        target_modules: Optional[list] = None,
        lora_dropout: float = 0.05,
    ):
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            use_gradient_checkpointing=False,
        )
        print(f"LoRA applied (r={r}, alpha={lora_alpha})")

    def train(
        self,
        train_data: list,
        eval_data: Optional[list] = None,
        num_epochs: int = 3,
        batch_size: int = 1,
        learning_rate: float = 2e-4,
        gradient_accumulation_steps: int = 8,
        max_seq_length: int = 1024,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        lr_scheduler_type: str = "cosine",
        logging_steps: int = 10,
        save_steps: int = 400,
        eval_steps: int = 1000,
        val_batches: int = 32,
    ):
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data) if eval_data else None

        config = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            max_seq_length=max_seq_length,
            dataset_text_field="text",
            packing=False,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=2,
            val_batches=val_batches if eval_data else 0,
            steps_per_eval=eval_steps if eval_data else 0,
        )

        trainer = SFTTrainer(
            model=self.model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        print("Starting training...")
        trainer.train()
        print("Training complete")

        # mlx-tune's save_model tries to fuse the base model, which is
        # unnecessary and slow on Mac. Save just the LoRA adapter instead.
        self.adapter_path = str(Path(self.output_dir) / "adapters")
        self.tokenizer.save_pretrained(self.adapter_path)
        print(f"Tokenizer saved to {self.adapter_path}")

    def load_for_inference(self, adapter_path: Optional[str] = None):
        adapter_path = adapter_path or self.adapter_path
        from mlx_lm import load
        self.model, self.tokenizer = load(
            self.base_model_name,
            adapter_path=adapter_path,
        )
        print(f"Inference model loaded from {adapter_path}")

    def inference(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        from clean_stream import clean_text
        messages = [
            {"role": "system", "content": "You are an Indian fintech regulatory expert. Answer clearly and accurately."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        out = generate(
            self.model,
            self.tokenizer,
            prompt=text,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature, top_p=0.9),
            verbose=False,
        ).strip()
        return clean_text(out)
