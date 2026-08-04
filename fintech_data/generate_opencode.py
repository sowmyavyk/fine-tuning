import json
import re
import subprocess
import random
from pathlib import Path

from tqdm import tqdm

from .config import FINTECH_TOPICS


class OpencodeDataGenerator:
    def __init__(self, output_dir: str = "data", model: str = "opencode/deepseek-v4-flash-free", workdir: str = None, timeout: int = 3600):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.workdir = workdir
        self.timeout = timeout

    def _build_question_pool(self, num_pairs: int, batch_size: int = 10):
        all_questions = []
        for domain in FINTECH_TOPICS:
            all_questions.extend(
                {"question": q, "domain": domain["domain"]}
                for q in domain["subtopics"]
            )
        pool = []
        cycles = max(1, num_pairs // len(all_questions) + 1)
        for c in range(cycles):
            random.shuffle(all_questions)
            pool.extend(all_questions)
        pool = pool[:num_pairs]
        return [pool[i : i + batch_size] for i in range(0, len(pool), batch_size)]

    def _run_prompt(self, prompt: str) -> str:
        cmd = ["opencode", "run", prompt, "-m", self.model]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=self.workdir, timeout=self.timeout
        )
        return result.stdout.strip()

    @staticmethod
    def _parse_json_array(text: str) -> list:
        if not text:
            return []
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def _build_batch_prompt(self, batch: list) -> str:
        numbered = "\n".join(
            f"{i + 1}. {item['question']}" for i, item in enumerate(batch)
        )
        return (
            f"You are an Indian fintech regulatory expert. Answer each question below "
            f"in 2-4 clear sentences for a newcomer to fintech. Be accurate.\n\n"
            f"Return ONLY a valid JSON array of exactly {len(batch)} objects, "
            f"each with keys \"question\" (the exact question text) and \"answer\" "
            f"(your 2-4 sentence explanation), in the same order as the questions.\n"
            f"No markdown, no commentary, no code fences.\n\n"
            f"Questions:\n{numbered}"
        )

    def generate(
        self, num_pairs: int = 750, batch_size: int = 10,
        checkpoint_every: int = 50, checkpoint_filename: str = None,
        source_label: str = None,
    ) -> list:
        source_label = source_label or "opencode_" + self.model.split("/")[-1]
        existing = {}
        if checkpoint_filename:
            for p in self.load(checkpoint_filename):
                existing[p["instruction"]] = p

        batches = self._build_question_pool(num_pairs, batch_size)
        pairs = list(existing.values())
        done_questions = set(existing.keys())

        for batch in tqdm(batches, desc=source_label):
            new_batch = [b for b in batch if b["question"] not in done_questions]
            if not new_batch:
                continue

            prompt = self._build_batch_prompt(new_batch)
            try:
                output = self._run_prompt(prompt)
            except subprocess.TimeoutExpired:
                print("  batch timed out - will retry on next run")
                continue

            answers = self._parse_json_array(output)
            by_q = {
                str(a.get("question")).strip(): str(a.get("answer", "")).strip()
                for a in answers if isinstance(a, dict) and a.get("answer")
            }
            batch_added = 0
            for item in new_batch:
                answer = by_q.get(item["question"].strip())
                if not answer or len(answer) < 20:
                    continue
                pairs.append({
                    "instruction": item["question"],
                    "response": answer,
                    "domain": item["domain"],
                    "source": source_label,
                })
                done_questions.add(item["question"])
                batch_added += 1

            if batch_added == 0:
                print("  no valid answers parsed this batch - skipping")

            if checkpoint_filename:
                self.save(pairs, checkpoint_filename)

        return pairs

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
