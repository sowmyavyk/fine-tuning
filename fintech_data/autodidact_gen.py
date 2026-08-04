import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

from data_processor import DataProcessor

SYSTEM_PROMPT = """You are a fintech compliance tutor grounded in Indian regulatory documents (PMLA 2002, PML(Maintenance of Records) Rules 2005, FIU-IND enforcement orders, SEBI documents).
Read the source passage carefully. Generate THREE high-quality, exam-style questions and answers, ALL strictly grounded in the passage text — do NOT add outside knowledge.

Requirements:
- Each question must require reasoning ABOUT the passage (interpretation, application, "what does this mean for a reporting entity"), not a trivia lookup.
- Each answer must be clear and well-structured (2-5 sentences) directly referencing the passage content.
- Output ONLY valid JSON with the form: {"pairs": [{"question": "...", "answer": "..."}, ...]} with exactly 3 pairs.
- Do not include the source text in the output.
"""


class AutoDidactGenerator:
    """AutoDidact-style: teacher LLM reads source doc chunks, writes grounded Q&A."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        max_workers: int = 1,
        cache_file: str = "data/fintech_data_autodidact.json",
    ):
        self.model = model
        self.cache_file = Path(cache_file)
        self.processor = DataProcessor()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.done = self._load_progress()

    def _load_progress(self) -> dict:
        if self.cache_file.exists():
            with open(self.cache_file, encoding="utf-8") as f:
                return {"pairs": json.load(f)}
        return {"pairs": []}

    def _save(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self._done_pairs(), f, indent=1, ensure_ascii=False)

    def _done_pairs(self) -> list:
        return self.done["pairs"]

    @staticmethod
    def _chunk_doc(text: str, chunk_size: int = 900, overlap: int = 100) -> list:
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) <= chunk_size:
            return [text] if len(text) > 150 else []
        chunks = []
        start = 0
        while start < len(text):
            chunk = text[start : start + chunk_size]
            # cut at sentence boundary if possible
            candidates = [
                chunk.rfind(". "),
                chunk.rfind(";"),
                chunk.rfind("\n\n"),
            ]
            valid = [c for c in candidates if c > chunk_size // 2]
            if valid:
                cut = max(valid)
                chunk = chunk[: cut + 1]
            chunks.append(chunk.strip())
            if len(chunk.strip()) < 150:
                break
            start += len(chunk) - overlap
        return [c for c in chunks if len(c) >= 150]

    def _generate_one(self, chunk: str, source_label: str, url: str) -> list:
        """Generate up to 3 grounded Q&A pairs from one chunk."""
        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"SOURCE PASSAGE:\n{chunk}\n\nGenerate the Q&A pairs as JSON."},
                    ],
                    temperature=0.5,
                    max_tokens=1200,
                )
                text = resp.choices[0].message.content.strip()
                text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
                parsed = json.loads(text)
                raw_pairs = parsed.get("pairs", [])
                if isinstance(parsed, list):
                    raw_pairs = parsed
                out = []
                for item in raw_pairs:
                    question = str(item.get("question", "")).strip()
                    answer = str(item.get("answer", "")).strip()
                    if not question or not answer:
                        continue
                    if self.processor.is_gibberish(answer):
                        continue
                    out.append({
                        "instruction": question,
                        "response": answer,
                        "domain": "autodidact_grounded",
                        "source": "autodidact",
                        "source_doc": source_label,
                        "url": url,
                        "model": self.model,
                    })
                return out
            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate_limit" in msg.lower():
                    wait = min(60, 5 * (2 ** attempt))
                    print(f"    rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                print(f"    gen ERR: {msg[:80]}")
                return []
        return []

    def generate_from_docs(
        self,
        docs: list,
        target_pairs: int = 10000,
        retries: int = 2,
    ) -> list:
        """Chunk all docs, generate Q&A per chunk until target reached."""
        all_chunks = []
        for doc in docs:
            for chunk in self._chunk_doc(doc["text"]):
                all_chunks.append((chunk, doc["title"], doc.get("url", "")))
        random.Random(42).shuffle(all_chunks)

        produced = len(self._done_pairs())
        print(f"Source chunks available: {len(all_chunks)}")
        print(f"Already produced: {produced} / target {target_pairs}")
        if produced >= target_pairs:
            return self._done_pairs()

        idx = 0
        while produced < target_pairs and idx < len(all_chunks) * retries:
            chunk, title, url = all_chunks[idx % len(all_chunks)]
            idx += 1
            new_pairs = self._generate_one(chunk, title, url)
            for pair in new_pairs:
                self.done["pairs"].append(pair)
                produced += 1
                if produced % 50 == 0:
                    self._save()
                    print(f"  progress: {produced}/{target_pairs}")
            time.sleep(1.0)

        self._save()
        print(f"Done: {produced} grounded pairs")
        return self._done_pairs()


if __name__ == "__main__":
    import json as _json

    with open("data/autodidact_source_docs.json") as f:
        source = _json.load(f)
    all_docs = []
    for k, docs in source.items():
        all_docs.extend(docs)

    target = int(os.getenv("AUTODIDACT_TARGET", "10000"))
    gen = AutoDidactGenerator()
    pairs = gen.generate_from_docs(all_docs, target_pairs=target)
    print(f"Final: {len(pairs)} pairs saved to {gen.cache_file}")
