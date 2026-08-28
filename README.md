# FinLens 🪙

**An Indian fintech compliance Q&A model** — fine-tuned on real RBI, SEBI, FIU-IND, and Digital-India sources.

Ask it about CKYC, KYC, AML, VKYC, CERSAI, PMLA 2002, or Aadhaar eKYC and it answers like a compliance officer — grounded in actual Indian regulations.

## 🚀 Try it live

**https://web-jet-nine-12.vercel.app**

A live demo UI of the real fine-tuned model (Qwen2.5-1.5B + LoRA) with RAG-grounded answers and cited sources.

> **Note:** the model backend runs on free tier GPUs (Kaggle), which are session-based and spin down when idle. If the chat returns a 404, a backend session isn't running —
> start one with `deploy/kaggle/finlens_kaggle_backend.py` and point the UI's ⚙️ "Model backend URL" setting at the printed `trycloudflare.com` address.

## The problem

Legal/regulatory text is dense. Generic models hallucinate on "What is CKYC?" or "DKYC vs VKYC?". We want a small model that answers *correctly* and *grounded in real rules* — and runs on a laptop.

## What we built

```
💾  Data (44k+ Q&A pairs)
    ├─ RBI Master Directions (135 PDFs → DeepSeek-grounded Q&A)
    ├─ FIU-IND / SEBI / gov blogs (robots-polite crawler)
    ├─ DeepSeek-grounded Q&A (teacher LLM reads chunks → writes exam Q&A)
    └─ Pre-built Indian finance QA (HF dataset)

🤖  Model
    Qwen2.5-1.5B (4-bit) + LoRA → mlx-tune SFT
    ~1.5GB peak — trains on an 8GB MacBook M1
```

Every Q&A is tagged with its source doc + URL, so nothing is "made up".

## Pipeline

```bash
# 1. Fetch RBI Master Direction PDFs as raw source docs (free)
.venv/bin/python3.11 -c "
import sys; sys.path.insert(0,'fintech_data')
from build_20k_dataset import DatasetBuilder
DatasetBuilder().fetch_rbi_source_docs(max_pdfs=135)
"

# 2a. Generate grounded Q&A from FIU-IND / SEBI / gov blogs via DeepSeek
AUTODIDACT_TARGET=15000 .venv/bin/python3.11 fintech_data/autodidact_gen.py

# 2b. Generate grounded Q&A from RBI Master Directions (separate files, runs alongside 2a)
AUTODIDACT_TARGET=10000 \
AUTODIDACT_SOURCES=data/rbi_source_docs.json \
AUTODIDACT_CACHE=data/fintech_data_grounded_rbi.json \
AUTODIDACT_PROGRESS=data/autodidact_progress_rbi.json \
.venv/bin/python3.11 fintech_data/autodidact_gen.py

# 3. Crawl more gov sources (free, robots-polite)
.venv/bin/python3.11 fintech_data/gov_fetcher.py

# 4. Train & test → model.ipynb (cells 6 → 8 → 10)
```

## Stack

| Layer | Tool |
|---|---|
| Base model | `mlx-community/Qwen2.5-1.5B-4bit` |
| Fine-tuning | mlx-tune LoRA (r=16) |
| Data | `datasets`, pypdf, BeautifulSoup |
| Teacher LLM | DeepSeek `deepseek-chat` |

## Why it's trustworthy

- **Grounded** — answers reference actual regulatory text, not vibes
- **All grounded kept** — every DeepSeek-generated pair (FIU-IND + RBI) goes into training, capped only by the quality filter
- **Balanced** — caps apply to the *raw* sources (8k HF / 12k reliable) so generated data never gets drowned out
- **Source-attributed** — every pair tracks its origin doc + URL
- **Robots-polite** — crawls only sites that allow it

## Deployment (real model, free tier)

The real fine-tuned model is served with the same `serve.py` backend the UI talks to — RAG citation lookup + token-streaming SSE over `POST /api/chat`:

| Piece | Where | Cost |
|---|---|---|
| UI | Vercel (static) — **https://web-jet-nine-12.vercel.app** | free, always-on |
| Model backend | Kaggle GPU (`deploy/kaggle/finlens_kaggle_backend.py`) | free, 30h GPU/wk, session-based |
| Adapter weights | Hugging Face `Sowmyavyk/qwen-fintech-adapter` | free, storage only |
| RAG data + backend source | Hugging Face `Sowmyavyk/finlens-gguf` | free, storage only |

- The UI reads the backend URL at runtime from the ⚙️ settings (stored in `localStorage`), so you can re-point it at a fresh tunnel URL without redeploying.
- Run `deploy/kaggle/finlens_kaggle_backend.py` on a Kaggle GPU notebook, copy the printed tunnel URL, paste it into the UI settings.
- All backend docker/caddy/oracle self-host variants live under `deploy/`.

## Files

```
fintech_data/
  autodidact_gen.py     # DeepSeek teacher → grounded Q&A (parallel jobs via env vars)
  build_20k_dataset.py  # RBI PDF fetch (source docs + chunk pairs) + HF finance QA
  gov_fetcher.py        # robots-checked gov crawler
  data_processor.py     # filter / format / split
  finetune.py           # LoRA trainer + inference
model.ipynb             # merge → train → test
```

See `DATA_SOURCES.md` for the full source breakdown.
