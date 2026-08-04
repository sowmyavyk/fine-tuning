import json
import random
import re
from pathlib import Path
from typing import Optional


class DataProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_gibberish(response: str) -> bool:
        text = response or ""
        if len(text) < 30:
            return True
        words = re.findall(r"[A-Za-z]+", text)
        if not words:
            return True
        repeated = max(
            (words.count(w) for w in set(words)), default=0
        )
        if len(words) and repeated / len(words) > 0.15:
            return True
        if "<think>" in text or "Describe in detail:" in text or "Explain in detail:" in text:
            return True
        return False

    def filter_quality(self, pairs: list) -> list:
        cleaned = []
        for p in pairs:
            if not self.is_gibberish(p.get("response", "")):
                cleaned.append(p)
        print(f"Quality filter: {len(pairs)} -> {len(cleaned)} kept")
        return cleaned

    def merge_sources(
        self,
        synthetic_data: list,
        regulatory_data: Optional[dict] = None,
    ) -> list:
        all_pairs = list(synthetic_data)

        if regulatory_data:
            regulatory_pairs = self._convert_regulatory_to_pairs(regulatory_data)
            all_pairs.extend(regulatory_pairs)
            print(f"Added {len(regulatory_pairs)} regulatory-derived pairs")

        return all_pairs

    def _convert_regulatory_to_pairs(self, reg_data: dict) -> list:
        pairs = []
        domain_defaults = {
            "rbi_notifications": "RBI Regulations",
            "cersai": "CERSAI",
            "sebi": "SEBI Regulations",
            "aml": "AML Compliance",
        }

        for source_key, items in reg_data.items():
            domain = domain_defaults.get(source_key, source_key)
            for item in items:
                if isinstance(item, dict):
                    instruction = (
                        f"What is the latest regulatory update from "
                        f"{item.get('title', '')[:100]}?"
                    )
                    snippet = item.get("snippet") or item.get("title", "")
                    pairs.append({
                        "instruction": instruction,
                        "response": (
                            f"Based on information from {item.get('source', source_key)}: "
                            f"{snippet}\n\n"
                            f"Source: {item.get('url', 'N/A')}"
                        ),
                        "domain": domain,
                        "source": source_key,
                    })
        return pairs

    def split_dataset(
        self, pairs: list, train_ratio: float = 0.85
    ) -> tuple:
        shuffled = list(pairs)
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * train_ratio)
        train = shuffled[:split_idx]
        eval = shuffled[split_idx:]
        print(f"Train: {len(train)}, Eval: {len(eval)}")
        return train, eval

    def format_for_training(
        self, pairs: list, model_type: str = "qwen"
    ) -> list:
        formatted = []
        for p in pairs:
            if model_type == "qwen":
                text = (
                    f"<|im_start|>user\n{p['instruction']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{p['response']}<|im_end|>"
                )
            elif model_type == "llama":
                text = (
                    f"[INST] {p['instruction']} [/INST] "
                    f"{p['response']}</s>"
                )
            else:
                text = (
                    f"Question: {p['instruction']}\n"
                    f"Answer: {p['response']}"
                )
            formatted.append({"text": text, **p})
        return formatted

    def save_split(self, train: list, eval: list, prefix: str = "fintech"):
        train_path = self.data_dir / f"{prefix}_train.json"
        eval_path = self.data_dir / f"{prefix}_eval.json"
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train, f, indent=2, ensure_ascii=False)
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval, f, indent=2, ensure_ascii=False)
        print(f"Saved train ({len(train)}) and eval ({len(eval)})")
        return train_path, eval_path

    def load_dataset(self, prefix: str = "fintech") -> tuple:
        train_path = self.data_dir / f"{prefix}_train.json"
        eval_path = self.data_dir / f"{prefix}_eval.json"
        train = json.load(open(train_path)) if train_path.exists() else []
        eval = json.load(open(eval_path)) if eval_path.exists() else []
        return train, eval
