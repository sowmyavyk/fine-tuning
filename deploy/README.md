# FinLens — Deployable

Self-contained deployment for the FinLens fintech compliance copilot: a
FastAPI SSE backend (open-source `transformers` + `PEFT`), the fine-tuned
Qwen2.5-1.5B LoRA adapter, RAG citations, and the Next.js frontend — all
open-source and Dockerized so it runs on any Linux VPS, cloud VM, or
Apple-Silicon host.

> Why not MLX in Docker? The current Mac stack (`mlx_lm`) only runs on
> Apple Silicon in-process. For a client-shareable, host-agnostic deploy we
> use the Hugging Face backend instead. Your Colab-trained adapter is already
> in HF PEFT format, so no retraining is needed — the same weights serve both.

## One-command run

```bash
cd deploy
docker compose up --build -d
```

Watch it start:

```bash
docker compose logs -f backend
```

Then open **http://localhost:3000**.

The base model (`Qwen/Qwen2.5-1.5B`, ~3.3 GB) downloads from Hugging Face on
first start and is cached in the `hf-cache` volume. The 74 MB LoRA adapter and
the RAG data ship inside the image.

## What's in the image

```
backend/
  serve.py              # FastAPI, same /api/chat SSE contract as the MLX server
  citation_lookup.py    # RAG: keyword index over grounded Q&A pairs
  clean_stream.py       # truncates degenerate/looped generation
  model/adapter/        # HF PEFT LoRA adapter (r=16, α=32, 7 modules)
  data/*.json           # grounded Q&A used for retrieval + citations
frontend/Dockerfile     # Next.js standalone build (web/)
```

## Architecture

```
Browser ──► :3000 Next.js (static, streams tokens via fetch → SSE)
                │
                ▼
              :8000 FastAPI /api/chat
                ├─ RAG: retrieve top-k grounded Pairs from data/*.json → context
                ├─ Qwen2.5-1.5B + LoRA (HF, device auto / cuda / cpu / mps)
                ├─ repetition_penalty 1.15, temp 0.4, max 1024 tokens
                └─ SSE: {"type":"token"} … {"type":"sources"} … [DONE]
```

### Environment variables

| Var | Backend | Default |
|---|---|---|
| `MODEL_NAME` | HF base model id | `Qwen/Qwen2.5-1.5B` |
| `DEVICE` | `auto` \| `cpu` \| `cuda` \| `mps` | `auto` |
| `MAX_TOKENS` | max generated tokens | `1024` |
| `TEMP` | sampling temperature | `0.4` |
| `ADAPTER_DIR` | LoRA adapter path (in-image) | `/app/model/adapter` |

Frontend: `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).

## Deploying somewhere public

### 1. Cheap VPS (recommended)
Any 2 GB+ Linux box (e.g. a free Oracle Cloud ARM instance or a $5 VPS):

```bash
git clone <repo> && cd deploy
ssh server 'docker compose up -d --build'
```
Add a reverse proxy (Caddy / nginx — both open source) for TLS, then point a
subdomain at `:3000`.

### 2. Hugging Face Spaces (fully hosted, open source)
Create a **Docker Space** with the FastAPI backend image (`serve.py` +
adapter) on their free CPU or paid GPU tier:
- Base image: `python:3.11`
- Entrypoint: `uvicorn serve:app --host 0.0.0.0 --port 7860`
- Mount `model/adapter` + `data/` from this repo
- Point `NEXT_PUBLIC_API_URL` at the Space's public `/` URL.

### 3. Cloudflare Tunnel (zero infra, Mac-hosted)
For an instant demo URL from your own machine:
```bash
cloudflared tunnel --url http://localhost:3000
```
(Free, open-source client; the Mac must stay on.)

## Switching hardware later

- **NVIDIA GPU** → set `DEVICE=cuda` (or swap the backend to vLLM for higher
  throughput — vLLM loads the same PEFT adapter).
- **Apple Silicon host** → `DEVICE=mps`; or keep using the original `server.py`
  MLX path for maximum speed on Mac.

## Reproduce locally (no Docker)

```bash
cd deploy/backend
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn serve:app --host 0.0.0.0 --port 8000          # terminal 1
cd ../.. && cd web && npm ci && npm run build && npm run start  # terminal 2
```