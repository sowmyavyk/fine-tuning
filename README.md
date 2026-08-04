# FinLens 🪙

**An Indian fintech compliance Q&A model** — fine-tuned on real RBI, SEBI, FIU-IND, and Digital-India sources.

Ask it about CKYC, KYC, AML, VKYC, CERSAI, PMLA 2002, or Aadhaar eKYC and it answers like a compliance officer — grounded in actual Indian regulations.

## The problem

Legal/regulatory text is dense. Generic models hallucinate on "What is CKYC?" or "DKYC vs VKYC?". We want a small model that answers *correctly* and *grounded in real rules* — and runs on a laptop.

## What we built

```
💾  Data (22k+ Q&A pairs)
    ├─ RBI Master Directions (PDFs, scraped)
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
# 1. Generate grounded Q&A from FIU/SEBI/blogs via DeepSeek
AUTODIDACT_TARGET=15000 .venv/bin/python3.11 fintech_data/autodidact_gen.py

# 2. Crawl more gov sources (free, robots-polite)
.venv/bin/python3.11 fintech_data/gov_fetcher.py

# 3. Train & test → model.ipynb (cells 6 → 8 → 10)
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
- **Balanced** — no single source dominates the training mix
- **Source-attributed** — every pair tracks its origin doc + URL
- **Robots-polite** — crawls only sites that allow it

## Files

```
fintech_data/
  autodidact_gen.py     # DeepSeek teacher → grounded Q&A
  build_20k_dataset.py  # RBI PDFs + HF finance QA
  gov_fetcher.py        # robots-checked gov crawler
  data_processor.py     # filter / format / split
  finetune.py           # LoRA trainer + inference
model.ipynb             # merge → train → test
```

See `DATA_SOURCES.md` for the full source breakdown.
