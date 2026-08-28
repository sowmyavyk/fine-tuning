import json
import re
import unicodedata
from pathlib import Path

GROUNDED_FILES = [
    Path("data/fintech_data_grounded.json"),
    Path("data/fintech_data_grounded_rbi.json"),
]


def _tokenize(text: str) -> set:
    text = unicodedata.normalize("NFKD", text or "")
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class SourceIndex:
    """Builds an in-memory keyword index over the grounded pairs.

    Each pair is a real Q&A generated from an actual regulatory doc, so its
    `source_doc` + `url` are the citations we surface to the user.
    """

    STOP = {
        "what", "is", "are", "the", "a", "an", "of", "to", "in", "for", "on",
        "and", "or", "do", "does", "how", "why", "when", "which", "who",
        "from", "with", "by", "it", "this", "that", "not", "be", "as", "at",
        "can", "should", "me", "my", "whats", "im", "dont", "want",
    }

    def __init__(self, grounded_files: list = None):
        self.pairs = []
        for path in (grounded_files or GROUNDED_FILES):
            if not path.exists():
                print(f"NOTE: {path} not found, skipped")
                continue
            with open(path, encoding="utf-8") as f:
                self.pairs.extend(json.load(f))
        self._index = []
        for p in self.pairs:
            instruction = _tokenize(p.get("instruction", ""))
            response = _tokenize(p.get("response", ""))
            doc = _tokenize(p.get("source_doc", ""))
            self._index.append({
                "pair": p,
                "instruction": instruction,
                "response": response,
                "doc": doc,
            })
        print(f"SourceIndex ready: {len(self.pairs)} grounded pairs indexed")

    def search(self, query: str, k: int = 3) -> list:
        q = _tokenize(query) - self.STOP
        if not q:
            return []
        scored = []
        for item in self._index:
            score = 0.0
            overlap = q & item["instruction"]
            score += 2.0 * len(overlap) / max(len(q), 1)
            score += 1.0 * len(q & item["response"]) / max(len(q), 1)
            score += 1.5 * len(q & item["doc"]) / max(len(q), 1)
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]

    def citation(self, item: dict) -> dict:
        p = item["pair"]
        return {
            "source_doc": p.get("source_doc", "Unknown document"),
            "url": p.get("url", "N/A"),
            "snippet": p.get("response", "")[:180],
        }


def build_index(grounded_files: list = None) -> SourceIndex:
    return SourceIndex(grounded_files)


if __name__ == "__main__":
    import sys

    idx = build_index()
    q = sys.argv[1] if len(sys.argv) > 1 else "What is VKYC?"
    print(f"Query: {q}\n")
    for item in idx.search(q):
        c = idx.citation(item)
        print(f"  • {c['source_doc']}\n    {c['url']}\n    {c['snippet']}\n")
