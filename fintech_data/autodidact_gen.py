import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Optional

import requests
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

from data_processor import DataProcessor

DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Compact system prompt — keep token overhead tiny.
SYSTEM_PROMPT = (
    "You are a fintech compliance tutor grounded in Indian regulatory documents "
    "(PMLA 2002, PML(Maintenance of Records) Rules 2005, FIU-IND orders, SEBI). "
    "From the passage, write 3 exam Q&A pairs grounded ONLY in the passage text. "
    "Answers 2-5 sentences, structured, reasoning-based. "
    'Respond with ONLY JSON: {"pairs":[{"question":"...","answer":"..."},"..."]}. '
    "Do not include the source text or any explanation outside JSON."
)

# Indian finance QA dataset from HuggingFace — loaded directly (no generation needed)
HF_DATASET_NAME = "Priyanshu-24/adaption-indian-finance-qa"

PROGRESS_FILE = "data/autodidact_progress.json"


class AutoDidactGenerator:
    """AutoDidact-style: teacher LLM reads source doc chunks, writes grounded Q&A.

    Uses the DeepSeek direct API (no HF router). Protects token spend with:
      1. chunk-hash cache  — a chunk already turned into Q&A is never re-sent;
      2. prompt trimming   — compact system prompt, no chat history;
      3. JSON-only mode    — response_format json_object, no markdown fences;
      4. bounded retries   — 429/5xx backoff, no infinite loops;
      5. pair dedupe       — identical (question,answer) never duplicated;
      6. dry-run estimate  — prints projected tokens/cost before calling.
    """

    def __init__(
        self,
        model: str = DEEPSEEK_MODEL,
        max_workers: int = 1,
        cache_file: str = "data/fintech_data_grounded.json",
        max_regen: int = 3,
        progress_file: Optional[str] = None,
    ):
        self.model = model
        self.cache_file = Path(cache_file)
        self.progress_file = Path(progress_file or os.getenv("AUTODIDACT_PROGRESS", PROGRESS_FILE))
        self.processor = DataProcessor()
        self.max_regen = max_regen
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is missing from .env")
        self.done = self._load_progress()
        self._seen = {(p["instruction"], p["response"]) for p in self._done_pairs()}

    # ------------------------------------------------------------------ io --
    def _load_progress(self) -> dict:
        pairs = []
        if self.cache_file.exists():
            with open(self.cache_file, encoding="utf-8") as f:
                try:
                    pairs = json.load(f)
                except json.JSONDecodeError:
                    pairs = []
        if self.progress_file.exists():
            with open(self.progress_file, encoding="utf-8") as f:
                try:
                    cc = json.load(f)
                    cc.setdefault("mined", {})
                    cc.setdefault("failed", {})
                    cc.setdefault("done", [])
                    # backward-compat: migrate old "done" list into mined counts
                    for cid in cc["done"]:
                        cc["mined"].setdefault(cid, 1)
                    self.chunk_cache = cc
                    return {"pairs": pairs, "chunk_cache": cc}
                except json.JSONDecodeError:
                    pass
        self.chunk_cache = {"done": [], "mined": {}, "failed": {}}
        return {"pairs": pairs, "chunk_cache": self.chunk_cache}

    def _save(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self._done_pairs(), f, indent=1, ensure_ascii=False)
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(self.chunk_cache, f, indent=1, ensure_ascii=False)

    def _done_pairs(self) -> list:
        return self.done["pairs"]

    @staticmethod
    def _chunk_id(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    # --------------------------------------------------------------- chunks --
    @staticmethod
    def _chunk_doc(text: str, chunk_size: int = 900, overlap: int = 100) -> list:
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) <= chunk_size:
            return [text] if len(text) > 150 else []
        chunks = []
        start = 0
        while start < len(text):
            chunk = text[start : start + chunk_size]
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

    # --------------------------------------------------------------- call ----
    def _call_deepseek(self, prompt_tokens_est: int, chunk: str) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"SOURCE PASSAGE:\n{chunk}\n\nGenerate the Q&A pairs as JSON."},
            ],
            "temperature": 0.5,
            "max_tokens": 900,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        for attempt in range(4):
            try:
                resp = requests.post(
                    f"{DEEPSEEK_BASE}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=90,
                )
            except requests.RequestException as e:
                print(f"    net ERR: {str(e)[:60]}; retry {attempt+1}/4")
                time.sleep(3 * (attempt + 1))
                continue

            code = resp.status_code
            if code == 200:
                return resp.json()
            if code == 401:
                print("    FATAL: DEEPSEEK_API_KEY invalid/revoked")
                raise RuntimeError("DeepSeek auth failed (401)")
            if code == 402:
                print("    FATAL: DeepSeek balance depleted (402)")
                raise RuntimeError("DeepSeek insufficient balance (402) — top up at https://platform.deepseek.com")
            if code == 429:
                wait = min(90, 8 * (2 ** attempt))
                print(f"    rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if code >= 500:
                wait = 4 * (attempt + 1)
                print(f"    server {code}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            # 400 etc — log and bail on this chunk (won't retry forever)
            print(f"    gen ERR {code}: {resp.text[:120]}")
            return {}
        return {}

    def _parse_pairs(self, data: dict) -> list:
        if not data:
            return []
        try:
            content = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            return []
        content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.M).strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if not m:
                return []
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                return []
        raw = parsed.get("pairs", []) if isinstance(parsed, dict) else parsed
        if not isinstance(raw, list):
            return []
        out = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            if not q or not a or len(a) < 40:
                continue
            if self.processor.is_gibberish(a):
                continue
            out.append({"instruction": q, "response": a})
        return out

    # ------------------------------------------------------------ generate --
    def _generate_one(self, chunk: str, source_label: str, url: str) -> list:
        """Generate Q&A from a chunk. Re-mining allowed up to `max_regen` times
        per chunk (same chunk, fresh generation -> more grounded variety)."""
        cid = self._chunk_id(chunk)
        mined = self.chunk_cache["mined"].get(cid, 0)
        if mined >= self.max_regen:
            return []
        if self.chunk_cache["failed"].get(cid, 0) >= 2:
            return []

        data = self._call_deepseek(400, chunk)
        pairs = self._parse_pairs(data)
        if pairs:
            self.chunk_cache["mined"][cid] = mined + 1
        else:
            self.chunk_cache["failed"][cid] = self.chunk_cache["failed"].get(cid, 0) + 1
        return pairs

    def generate_from_docs(
        self,
        docs: list,
        target_pairs: int = 10000,
        retries: int = 2,
    ) -> list:
        """Chunk all docs, generate Q&A per chunk until target reached.

        Cached chunks (already turned into Q&A) are skipped without any API call.
        """
        all_chunks = []
        for doc in docs:
            for chunk in self._chunk_doc(doc["text"]):
                all_chunks.append((chunk, doc["title"], doc.get("url", "")))
        random.Random(42).shuffle(all_chunks)

        produced = len(self._done_pairs())
        remaining = target_pairs - produced

        cache_done = len(self.chunk_cache["mined"])
        cache_failed = len(self.chunk_cache["failed"])
        todo_chunks = []
        for chunk, title, url in all_chunks:
            cid = self._chunk_id(chunk)
            if self.chunk_cache["mined"].get(cid, 0) >= self.max_regen:
                continue
            if self.chunk_cache["failed"].get(cid, 0) >= 2:
                continue
            todo_chunks.append((chunk, title, url))

        print(f"Source chunks available: {len(all_chunks)}")
        print(f"Cache: {cache_done} mined, {cache_failed} failed (max_regen={self.max_regen})")
        print(f"Chunks left to process: {len(todo_chunks)}")
        print(f"Pairs so far (DeepSeek only): {produced} / target {target_pairs} ({remaining} needed)")
        if produced >= target_pairs:
            print("Target already reached — nothing to generate.")
            return self._done_pairs()
        if not todo_chunks:
            print("No unprocessed chunks left — re-running yields nothing new.")
            return self._done_pairs()

        # Dry-run budget estimate (rough: ~140 prompt + ~350 completion tokens/call)
        est_calls = min(len(todo_chunks), (remaining + 2) // 3)
        est_in = est_calls * 140 / 1_000_000
        est_out = est_calls * 350 / 1_000_000
        print(f"\n=== DRY RUN BUDGET ===")
        print(f"  ~{est_calls} API calls (3 pairs each)")
        print(f"  ~{est_in:.3f}M in + ~{est_out:.3f}M out tokens")
        print(f"  DeepSeek (deepseek-chat ~$0.27/$1.10 per M): ~${est_in*0.27 + est_out*1.10:.2f}")
        proceed = input("  Proceed? [y/N]: ").strip().lower()
        if proceed != "y":
            print("Aborted by user.")
            return self._done_pairs()

        idx = 0
        calls = 0
        while produced < target_pairs and calls < len(todo_chunks) * retries:
            chunk, title, url = todo_chunks[idx % len(todo_chunks)]
            idx += 1
            calls += 1
            new_pairs = self._generate_one(chunk, title, url)
            added = 0
            for p in new_pairs:
                key = (p["instruction"], p["response"])
                if key not in self._seen:
                    self.done["pairs"].append({
                        **p,
                        "domain": "autodidact_grounded",
                        "source": "autodidact",
                        "source_doc": title,
                        "url": url,
                        "model": self.model,
                    })
                    self._seen.add(key)
                    produced += 1
                    added += 1
            if produced % 50 == 0:
                self._save()
                print(f"  progress: {produced}/{target_pairs} (this round +{added})")
            time.sleep(0.5)

        self._save()
        print(f"Done: {produced} grounded pairs (chunks mined: {len(self.chunk_cache['mined'])})")
        return self._done_pairs()

    # ----------------------------------------------------------- hf dataset --
    def load_hf_dataset(self, target_pairs: int = 10000, include_cats: Optional[set] = None) -> list:
        """Load pre-made Indian finance QA from a HF dataset directly (free)."""
        include_cats = include_cats or {
            "banking_regulation", "capital_markets", "digital_payments",
        }
        print(f"Loading HF dataset: {HF_DATASET_NAME}")
        ds = load_dataset(HF_DATASET_NAME, split="train")

        pairs = []
        count = 0
        for row in ds:
            count += 1
            cat = row.get("category", "")
            src = row.get("source", "")
            inp = row.get("input", "")
            out = row.get("output", "")
            if len(out) < 50:
                continue
            if cat not in include_cats and src != "RBI":
                continue
            pairs.append({
                "instruction": str(inp).strip(),
                "response": str(out).strip(),
                "domain": f"hf_indian_finance_{cat or 'other'}",
                "source": "hf_adaption_indian_finance",
                "source_doc": f"HF:{HF_DATASET_NAME}",
                "url": "",
                "model": "hf_dataset",
            })
            if len(pairs) >= target_pairs:
                break

        print(f"Loaded {len(pairs)} HF pairs (scanned {count} rows)")
        return pairs


def load_source_docs(*filenames: str) -> list:
    """Load docs from one or more source-doc JSON files into a flat list."""
    import json as _json

    all_docs = []
    for fn in filenames:
        path = Path(fn)
        if not path.exists():
            print(f"NOTE: source file {fn} not found, skipping")
            continue
        with open(path, encoding="utf-8") as f:
            data = _json.load(f)
        if isinstance(data, dict):
            for k, docs in data.items():
                all_docs.extend(docs)
        elif isinstance(data, list):
            all_docs.extend(data)
    return all_docs


if __name__ == "__main__":
    sources = os.getenv("AUTODIDACT_SOURCES", "").split(",")
    if any(sources):
        all_docs = load_source_docs(*[s.strip() for s in sources if s.strip()])
    else:
        all_docs = load_source_docs(
            "data/autodidact_source_docs.json",
            "data/gov_blogs.json",
            "data/gov_source_docs.json",
        )
    print(f"Total source docs: {len(all_docs)}")

    target = int(os.getenv("AUTODIDACT_TARGET", "10000"))
    max_regen = int(os.getenv("AUTODIDACT_MAX_REGEN", "3"))
    cache_file = os.getenv("AUTODIDACT_CACHE", "data/fintech_data_grounded.json")
    gen = AutoDidactGenerator(max_regen=max_regen, cache_file=cache_file)

    # AutoDidact: DeepSeek generates grounded QA from FIU-IND + SEBI + blogs.
    # Counts ONLY DeepSeek-generated pairs (domain=autodidact_grounded) toward
    # the target. HF dataset rows live in a separate file, never reloaded here.
    gen.generate_from_docs(all_docs, target_pairs=target, retries=3)
    grounded = [p for p in gen.done["pairs"] if p.get("domain") == "autodidact_grounded"]
    print(f"Final: {len(grounded)} DeepSeek-grounded pairs in {gen.cache_file}")
