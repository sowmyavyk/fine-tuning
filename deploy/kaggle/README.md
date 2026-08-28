# FinLens Kaggle Deployment

Free GPU deployment on Kaggle (T4/P100, 30h/week, no card needed).

## Quick Start (2 minutes)

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Enable GPU: **Settings** → **Accelerator** → **GPU T4 ×2** (or P100)
3. Add a new cell, paste the contents of `finlens_kaggle_quick.py`
4. Run the cell (Shift+Enter)
5. Wait ~3 minutes for model download + server startup
6. Copy the **Public URL** (trycloudflare.com)
7. Share with client or test with curl

## Test

```bash
curl -X POST https://YOUR-URL.trycloudflare.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is PMLA?"}],"stream":true}'
```

## Connect Frontend

Set `API_URL` to the Cloudflare tunnel URL, or add to your `.env.local`:
```
NEXT_PUBLIC_API_URL=https://YOUR-URL.trycloudflare.com
```

## Limitations

- **Session-based**: Kaggle sessions expire after 8h (GPU) or 12h (CPU)
- **GPU quota**: Free tier = 30h/week. Each session costs ~1-2h
- **Restart**: When session expires, re-run the cell to get a new URL
- **Model**: 1.5B FinLens (merged base + LoRA adapter)

## What's Included

- `finlens-merged.gguf` (3.09 GB F16) — Qwen2.5-1.5B + LoRA adapter merged
- `finlens_kaggle_quick.py` — One-cell deployment script
- `finlens_kaggle.py` — Full build-from-source version (slower but more control)

## Architecture

```
Browser → Cloudflare Tunnel → llama.cpp server (Kaggle GPU)
                                    ↓
                            Qwen2.5-1.5B + LoRA adapter
                            (FinLens fine-tuned model)
```
