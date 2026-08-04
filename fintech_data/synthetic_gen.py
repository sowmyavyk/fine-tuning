import json
import time
import random
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from tqdm import tqdm

from .config import FINTECH_TOPICS, SYSTEM_PROMPT, INFERENCE_CONFIG


class SyntheticDataGenerator:
    def __init__(self, output_dir: str = "data", max_workers: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

    def _build_messages(self, topic: str) -> list:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Explain in detail: {topic}"},
        ]

    def _build_question_pool(self, num_pairs: int):
        all_questions = []
        for domain in FINTECH_TOPICS:
            all_questions.extend(
                {"question": q, "domain": domain["domain"]}
                for q in domain["subtopics"]
            )
        temps = [0.5, 0.7, 0.85, 1.0]
        pool = []
        cycles = max(1, num_pairs // len(all_questions) + 1)
        for c in range(cycles):
            random.shuffle(all_questions)
            for i, item in enumerate(all_questions):
                if len(pool) >= num_pairs:
                    break
                pool.append({
                    "question": item["question"],
                    "domain": item["domain"],
                    "temperature": temps[(i + c) % len(temps)],
                })
        return pool

    def _generate_qwen_single(self, model, tokenizer, item: dict) -> dict:
        messages = self._build_messages(item["question"])
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=INFERENCE_CONFIG["max_tokens"],
                temperature=item["temperature"],
                top_p=INFERENCE_CONFIG["top_p"],
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        return {
            "instruction": item["question"],
            "response": response,
            "domain": item["domain"],
            "source": "qwen_local",
        }

    def generate_with_qwen(
        self, model, tokenizer, num_pairs: int = 500, batch_size: int = 4
    ) -> list:
        import torch
        pool = self._build_question_pool(num_pairs)
        pairs = []

        for i in tqdm(range(0, len(pool), batch_size), desc="Qwen local"):
            batch = pool[i : i + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    self._build_messages(item["question"]),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for item in batch
            ]
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True
            ).to(model.device)
            temp = batch[0]["temperature"]

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=INFERENCE_CONFIG["max_tokens"],
                    do_sample=True,
                    temperature=temp,
                    top_p=INFERENCE_CONFIG["top_p"],
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            for j, item in enumerate(batch):
                response = tokenizer.decode(
                    output[j][inputs.input_ids.shape[1]:], skip_special_tokens=True
                ).strip()
                if response:
                    pairs.append({
                        "instruction": item["question"],
                        "response": response,
                        "domain": item["domain"],
                        "source": "qwen_local",
                    })

        return pairs

    def _call_api_single(
        self, client_factory, model_name: str, item: dict, source_label: str,
        max_retries: int = 4, reasoning_effort: str = None,
    ) -> dict:
        client = client_factory()
        messages = self._build_messages(item["question"])

        for attempt in range(max_retries):
            try:
                extra = {}
                if reasoning_effort:
                    extra["reasoning_effort"] = reasoning_effort
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=item["temperature"],
                    max_tokens=INFERENCE_CONFIG.get("api_max_tokens", 512),
                    top_p=INFERENCE_CONFIG["top_p"],
                    **extra,
                )
                response = resp.choices[0].message.content.strip()
                response = self._strip_think_block(response)
                if response:
                    return {
                        "instruction": item["question"],
                        "response": response,
                        "domain": item["domain"],
                        "source": source_label,
                    }
                return None
            except Exception as e:
                message = str(e)
                lower = message.lower()
                # Retry on rate limits (429) AND intermittent auth errors (401)
                # Kimi/Moonshot sometimes returns 401 as a disguised rate limit.
                is_rate_limit = "429" in message or "rate_limit" in lower
                is_auth_401 = "401" in message or "invalid authentication" in lower or "invalid_authentication" in lower
                if is_rate_limit or is_auth_401:
                    retry_after = self._parse_retry_after(message)
                    wait = max(retry_after, 1.0) * (attempt + 1)
                    reason = "Rate limited" if is_rate_limit else "Auth/401"
                    print(f"  {reason} ({wait:.1f}s) at attempt {attempt+1} - retrying")
                    time.sleep(wait)
                    continue
                print(f"  API error: {message}")
                time.sleep(1.0)
                break
        return None

    @staticmethod
    def _strip_think_block(text: str) -> str:
        while "<think>" in text and "</think>" in text:
            start = text.find("<think>")
            end = text.find("</think>") + len("</think>")
            text = (text[:start] + text[end:]).strip()
        return text.strip()

    @staticmethod
    def _parse_retry_after(message: str) -> float:
        import re
        m = re.search(r"in\s+([\d.]+)\s*(ms|s)", message)
        if m:
            value = float(m.group(1))
            return value / 1000.0 if m.group(2) == "ms" else value
        return 2.0

    def _generate_via_api(
        self, client_factory, model: str, num_pairs: int, source_label: str,
        checkpoint_every: int = 50, checkpoint_filename: str = None,
        reasoning_effort: str = None,
    ) -> list:
        pool = self._build_question_pool(num_pairs)
        pairs = []
        fn = partial(self._call_api_single, client_factory, model,
                     source_label=source_label, reasoning_effort=reasoning_effort)

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(fn, item) for item in pool]
            for f in tqdm(as_completed(futures), total=len(futures), desc=source_label):
                result = f.result()
                if result:
                    pairs.append(result)
                    if checkpoint_filename and len(pairs) % checkpoint_every == 0:
                        self.save(pairs, checkpoint_filename)

        return pairs

    def generate_with_openai_compatible(
        self, api_key: str, base_url: str, model_name: str,
        num_pairs: int = 500, source_label: str = "openai_api",
        checkpoint_every: int = 50, checkpoint_filename: str = None,
        reasoning_effort: str = None,
    ) -> list:
        from openai import OpenAI
        def factory():
            return OpenAI(api_key=api_key, base_url=base_url)
        return self._generate_via_api(
            factory, model_name, num_pairs, source_label,
            checkpoint_every=checkpoint_every,
            checkpoint_filename=checkpoint_filename,
            reasoning_effort=reasoning_effort,
        )

    def generate_with_groq(
        self, api_key: str, num_pairs: int = 500,
        checkpoint_every: int = 50, checkpoint_filename: str = None,
    ) -> list:
        from groq import Groq
        def factory():
            return Groq(api_key=api_key)
        return self._generate_via_api(
            factory, "llama-3.3-70b-versatile", num_pairs, "groq_llama3_70b",
            checkpoint_every=checkpoint_every,
            checkpoint_filename=checkpoint_filename,
        )

    def generate_with_groq_qwen(
        self, api_key: str, num_pairs: int = 500,
        checkpoint_every: int = 50, checkpoint_filename: str = None,
    ) -> list:
        from groq import Groq
        def factory():
            return Groq(api_key=api_key)
        return self._generate_via_api(
            factory, "qwen/qwen3.6-27b", num_pairs, "groq_qwen3_6",
            checkpoint_every=checkpoint_every,
            checkpoint_filename=checkpoint_filename,
            reasoning_effort="none",
        )

    def save(self, pairs: list, filename: str = "fintech_data.json"):
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(pairs)} pairs to {path}")
        return path

    def load(self, filename: str = "fintech_data.json") -> list:
        path = self.output_dir / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
