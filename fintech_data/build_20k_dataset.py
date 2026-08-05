import json
import random
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from datasets import load_dataset
from pypdf import PdfReader

from data_processor import DataProcessor


FINANCE_KEYWORDS = [
    "kyc", "aml", "ckyc", "dkvc", "vkvc", "cersai", "rbi", "sebi",
    "compliance", "regulatory", "anti-money", "sanction", "banking",
    "pmla", "rupay", "upi", "nbfc", "lending", "credit", "loan",
    "collateral", "fintech", "deposit", "treasury", "capital",
    "risk", "audit", "fraud", "customer due diligence",
]


class DatasetBuilder:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processor = DataProcessor(data_dir)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })

    def _is_relevant(self, text: str) -> bool:
        lower = text.lower()
        return any(k in lower for k in FINANCE_KEYWORDS)

    def load_hf_finance(self, target: int = 8000) -> list:
        """RakeshMadasani banking-finance QA (clean format, global)."""
        pairs = []
        ds = load_dataset(
            "RakeshMadasani/banking-finance-qa-dataset", split="train", streaming=True
        )
        for row in ds:
            instruction = row.get("instruction", "")
            output = row.get("output", "")
            combined = instruction + " " + output
            if self._is_relevant(combined):
                pairs.append({
                    "instruction": instruction.strip(),
                    "response": output.strip(),
                    "domain": "finance_qa",
                    "source": "hf_rakesh_banking_finance",
                })
            if len(pairs) >= target:
                break
        print(f"HF RakeshMadasani: {len(pairs)} relevant pairs")
        return pairs

    def load_hf_glm(self, target: int = 12000) -> list:
        """GLM-5.2-Finance-80k deep CoT (strip <think> blocks)."""
        pairs = []
        ds = load_dataset(
            "ianncity/GLM-5.2-Finance-80000x", split="train", streaming=True
        )
        for row in ds:
            try:
                msgs = row["messages"]
                if isinstance(msgs, str):
                    msgs = json.loads(msgs.replace("'", '"'))
                user = next(
                    (m["content"] for m in msgs if m.get("role") == "user"), ""
                )
                assistant = next(
                    (m["content"] for m in msgs if m.get("role") == "assistant"), ""
                )
            except Exception:
                continue
            cleaned = re.sub(r"<think>.*?</think>", "", assistant, flags=re.S).strip()
            combined = user + " " + cleaned
            if self._is_relevant(combined) and len(cleaned) > 30:
                pairs.append({
                    "instruction": user.strip(),
                    "response": cleaned,
                    "domain": "finance_cot",
                    "source": "hf_glm_5.2_finance",
                })
            if len(pairs) >= target:
                break
        print(f"HF GLM-5.2-Finance: {len(pairs)} relevant pairs")
        return pairs

    def fetch_rbi_master_directions(self, max_pdfs: int = 25) -> list:
        """Download RBI master direction PDFs (real regulatory text)."""
        index_url = "https://www.rbi.org.in/Scripts/BS_ViewMasterDirections.aspx"
        resp = self.session.get(index_url, timeout=30)
        soup = BeautifulSoup(resp.text, "lxml")

        docs = []
        relevant_terms = [
            "kyc", "aml", "know your customer", "deposit", "credit",
            "lending", "customer", "information", "cyber", "risk",
            "governance", "fraud",
        ]
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "BS_ViewMasDirections" not in href:
                continue
            tr = a.find_parent("tr")
            pdf_link = None
            if tr:
                for pdf_a in tr.find_all("a", href=True):
                    if ".pdf" in pdf_a["href"].lower():
                        pdf_link = pdf_a["href"]
                        break
            title = a.get_text(strip=True)
            if not title or not pdf_link:
                continue
            if not any(t in title.lower() for t in relevant_terms):
                continue
            docs.append({"title": title, "url": pdf_link})
        print(f"Relevant RBI master direction PDFs: {len(docs)}")

        chunks = []
        for doc in docs[:max_pdfs]:
            try:
                r = self.session.get(doc["url"], timeout=60)
                r.raise_for_status()
                reader = PdfReader(BytesIO(r.content))
                text = "\n".join(
                    (pg.extract_text() or "") for pg in reader.pages
                )
                text = re.sub(r"\n{3,}", "\n\n", text).strip()
                # strip Devanagari/other non-Latin header garbage
                text = re.sub(r"[^\x00-\x7F]+", "", text)
                text = re.sub(r"\n{3,}", "\n", text)
                if len(text) < 500:
                    print(f"  skip (thin) {doc['title'][:50]}")
                    continue
                chunks.append({
                    "title": doc["title"],
                    "url": doc["url"],
                    "text": text[:40000],
                })
                print(f"  fetched {doc['title'][:50]}: {len(text)} chars")
            except Exception as e:
                print(f"  error {doc['title'][:40]}: {e}")
            time.sleep(0.5)

        pairs = self._rbi_chunks_to_pairs(chunks)
        print(f"RBI master direction PDFs -> {len(pairs)} chunk pairs")
        return pairs

    def fetch_rbi_source_docs(self, max_pdfs: int = 135, out_file: str = "data/rbi_source_docs.json") -> list:
        """Download RBI master direction PDFs as RAW source docs (for DeepSeek).

        Produces grounded Q&A later via autodidact_gen. Runs independently of
        the running DeepSeek job (writes only to out_file).
        """
        import json as _json

        index_url = "https://www.rbi.org.in/Scripts/BS_ViewMasterDirections.aspx"
        resp = self.session.get(index_url, timeout=30)
        soup = BeautifulSoup(resp.text, "lxml")

        relevant_terms = [
            "kyc", "aml", "know your customer", "deposit", "credit",
            "lending", "customer", "information", "cyber", "risk",
            "governance", "fraud", "digital", "payment",
        ]
        docs = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "BS_ViewMasDirections" not in href:
                continue
            tr = a.find_parent("tr")
            pdf_link = None
            if tr:
                for pdf_a in tr.find_all("a", href=True):
                    if ".pdf" in pdf_a["href"].lower():
                        pdf_link = pdf_a["href"]
                        break
            title = a.get_text(strip=True)
            if not title or not pdf_link:
                continue
            if not any(t in title.lower() for t in relevant_terms):
                continue
            docs.append({"title": title, "url": pdf_link})
        print(f"Relevant RBI master direction PDFs: {len(docs)}")

        out = []
        for i, doc in enumerate(docs[:max_pdfs]):
            try:
                r = self.session.get(doc["url"], timeout=60)
                r.raise_for_status()
                reader = PdfReader(BytesIO(r.content))
                text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
                text = re.sub(r"[^\x00-\x7F]+", "", text)
                text = re.sub(r"\n{3,}", "\n", text).strip()
                if len(text) < 500:
                    continue
                out.append({
                    "title": f"RBI {doc['title'][:80]}",
                    "url": doc["url"],
                    "text": text[:50000],
                    "source": "rbi_master_directions",
                })
                print(f"  [{i+1}/{len(docs[:max_pdfs])}] {doc['title'][:50]}: {len(text)} chars")
            except Exception as e:
                print(f"  error {doc['title'][:40]}: {e}")
            time.sleep(0.5)

        path = Path(out_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"Saved {len(out)} RBI source docs to {path}")
        return out

    def _rbi_chunks_to_pairs(self, chunks: list) -> list:
        pairs = []
        for chunk in chunks:
            text = chunk["text"]
            text = re.sub(r"[ \t]+", " ", text)
            lines = [l.strip() for l in text.split("\n")]
            window = ""
            for line in lines:
                if not line:
                    if len(window) >= 250:
                        self._emit_pair(pairs, window, chunk)
                        window = ""
                    continue
                if re.fullmatch(r"(Table|Annex|Schedule|Chapter|Part)\b.*", line) and len(line) < 60:
                    if len(window) >= 250:
                        self._emit_pair(pairs, window, chunk)
                    window = ""
                    continue
                if len(window) + len(line) <= 650:
                    window = (window + "\n" + line).strip()
                else:
                    if len(window) >= 250:
                        self._emit_pair(pairs, window, chunk)
                    window = line
            if len(window) >= 250:
                self._emit_pair(pairs, window, chunk)
        return pairs

    @staticmethod
    def _emit_pair(pairs: list, section: str, chunk: dict):
        heading = section.split("\n")[0][:80]
        pairs.append({
            "instruction": (
                f"According to the RBI {chunk['title']}, what does the section "
                f"on '{heading}' state? Quote the relevant provision."
            ),
            "response": section,
            "domain": "rbi_regulatory",
            "source": "rbi_master_directions",
            "url": chunk["url"],
        })

    def build(self, targets: Optional[dict] = None) -> list:
        targets = targets or {"hf_rakesh": 3000, "hf_glm": 12000, "rbi": 4000}
        all_pairs = []

        if targets.get("hf_rakesh", 0):
            all_pairs.extend(
                self.load_hf_finance(target=targets["hf_rakesh"])
            )
        if targets.get("hf_glm", 0):
            all_pairs.extend(
                self.load_hf_glm(target=targets["hf_glm"])
            )
        if targets.get("rbi", 0):
            all_pairs.extend(self.fetch_rbi_master_directions())

        # dedupe + quality filter
        seen = set()
        unique = []
        for p in all_pairs:
            key = (p["instruction"][:120], p["response"][:120])
            if key in seen:
                continue
            seen.add(key)
            unique.append(p)

        cleaned = self.processor.filter_quality(unique)
        random.shuffle(cleaned)
        print(f"Final reliable base dataset: {len(cleaned)} pairs")
        return cleaned

    def save(self, pairs: list, filename: str = "fintech_data_reliable.json"):
        path = self.data_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(pairs)} pairs to {path}")


if __name__ == "__main__":
    builder = DatasetBuilder()
    pairs = builder.build()
    builder.save(pairs)
