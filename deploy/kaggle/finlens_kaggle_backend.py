"""
FinLens — REAL model + YOUR UI backend on Kaggle (free GPU)
===========================================================
Runs the real Fine-tuned FinLens model (Qwen2.5-1.5B + LoRA) on a free Kaggle
GPU, using the SAME backend (serve.py) your web/ UI already talks to — RAG
sources + token-streaming SSE. Exposed via a Cloudflare Quick Tunnel.

The printed URL is your backend's /api/chat endpoint. Point your FinLens UI at
it with NEXT_PUBLIC_API_URL=<url>.

KAGGLE SETUP:
  1. New Notebook -> Settings -> Accelerator -> GPU T4 x2 (or P100)
     NOTE: set Accelerator to GPU from the dropdown BEFORE running.
  2. Paste this whole file into a cell -> Run (Shift+Enter)
  3. Wait ~4-6 min (model download + load). Copy the PUBLIC URL.
  4. That URL is your model backend. Wire it into your UI.

FREE: 30h GPU/week, ~8h per session. Re-run the cell for a fresh URL.
Not hosted on Hugging Face — HF is only a download source for weights.
"""

import os
import re
import subprocess
import time
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-1.5B"                 # public base model
ADAPTER_REPO = "Sowmyavyk/qwen-fintech-adapter"  # public LoRA adapter
STORE_REPO = "Sowmyavyk/finlens-gguf"            # public: data/ + backend/
WORKING = Path("/kaggle/working")
ADAPTER_DIR = WORKING / "adapter"
PORT = 8000
MAX_TOKENS = 1024
TUNNEL_BIN = WORKING / "cloudflared"


def run(cmd, check=True, **kw):
    print(f"  → {cmd[:90]}")
    return subprocess.run(cmd, shell=True, check=check, **kw)


def main():
    print("=" * 62)
    print("  FinLens | REAL model + YOUR UI backend (free Kaggle GPU)")
    print("=" * 62)

    # 1 ─ deps
    print("\n[1/7] Installing deps (transformers, peft, fastapi...)")
    run("pip install -q transformers peft accelerate safetensors sentencepiece "
        "fastapi 'uvicorn[standard]' pydantic huggingface_hub > /dev/null 2>&1")

    # 2 ─ download adapter + data + backend source (public, no auth)
    print("\n[2/7] Downloading LoRA adapter + RAG data + backend source")
    run(f'python -c "from huggingface_hub import snapshot_download;'
        f" snapshot_download(repo_id='{ADAPTER_REPO}', local_dir='{ADAPTER_DIR}')\"")
    run(f'python -c "from huggingface_hub import snapshot_download;'
        f" snapshot_download(repo_id='{STORE_REPO}', local_dir='{WORKING}')\"")
    # 3 ─ cloudflared
    print("\n[3/7] Installing cloudflared")
    if not TUNNEL_BIN.exists():
        run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/"
            "cloudflared-linux-amd64 -O /kaggle/working/cloudflared")
        run("chmod +x /kaggle/working/cloudflared")

    # 4 ─ start backend (the real model loads on GPU)
    print("\n[4/7] Starting FinLens backend (Qwen 1.5B + LoRA on GPU)...")
    run(f"lsof -ti:{PORT} | xargs -r kill -9 2>/dev/null || true")
    time.sleep(1)
    env = dict(os.environ)
    env.update({
        "MODEL_NAME": BASE_MODEL,
        "ADAPTER_DIR": str(ADAPTER_DIR),
        "DATA_DIR": str(WORKING),
        "DEVICE": "cuda",
        "MAX_TOKENS": str(MAX_TOKENS),
    })
    log = open(WORKING / "backend.log", "w")
    backend = subprocess.Popen(
        "python backend/serve.py", shell=True, env=env,
        stdout=log, stderr=subprocess.STDOUT, cwd=str(WORKING),
    )

    print("  Waiting for model + adapter to load (first run ~3-5 min)...")
    if not _wait_ready(f"http://localhost:{PORT}/api/health", 600):
        log.close()
        print("\n  ERROR: backend failed to start. Log tail:")
        print(open(WORKING / "backend.log").read()[-2500:])
        return
    log.close()
    print("  ✅ Backend READY on :8000")
    try:
        import urllib.request, json
        h = urllib.request.urlopen(f"http://localhost:{PORT}/api/health", timeout=5).read().decode()
        print("  Health:", h[:220])
    except Exception:
        pass

    # 5 ─ tunnel
    print("\n[5/7] Starting Cloudflare Quick Tunnel...")
    tunnel = _start_tunnel()
    url = _wait_tunnel_url(tunnel)

    if url:
        print("\n" + "=" * 62)
        print("  ✅ PUBLIC BACKEND URL (your UI endpoint):")
        print(f"\n      {url}\n")
        print("  Test it:")
        print(f"    curl -N -X POST {url}/api/chat \\")
        print('      -H "Content-Type: application/json" \\')
        print('      -d \'{"message":"What is VKYC?"}\'')
        print(f"\n  Wire into YOUR FinLens UI:")
        print(f"    Vercel:  add env  NEXT_PUBLIC_API_URL={url}")
        print(f"    Local:   NEXT_PUBLIC_API_URL={url}")
        print("\n  The SSE token stream + RAG sources render in your UI.")
        print("  Re-run the cell anytime for a fresh URL (session-based).")
        print("=" * 62)
    else:
        print("  ⚠ Tunnel URL not captured — check the log lines above.")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass


def _wait_ready(url, timeout):
    import urllib.request, urllib.error
    for i in range(int(timeout / 5)):
        time.sleep(5)
        try:
            urllib.request.urlopen(url, timeout=5)
            return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            pass
    return False


def _start_tunnel():
    return subprocess.Popen(
        [str(TUNNEL_BIN), "tunnel", "--url", f"http://localhost:{PORT}",
         "--protocol", "http2", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )


def _wait_tunnel_url(proc):
    pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    for _ in range(90):
        time.sleep(2)
        try:
            line = proc.stdout.readline().decode(errors="replace")
        except Exception:
            break
        m = pat.search(line)
        if m:
            return m.group(0)
    return None


if __name__ == "__main__":
    main()
