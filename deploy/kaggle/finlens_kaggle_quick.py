"""
FinLens Kaggle - Quick Start
=============================
Pre-built binaries, no compilation needed. Just run in a Kaggle notebook.

Setup:
1. New Kaggle notebook → GPU accelerator (T4)
2. Paste this script → Run
3. Get public URL → Share with client
"""

import os
import subprocess
import time
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
GGUF_REPO = "Sowmyavyk/finlens-gguf"
GGUF_FILE = "finlens-merged.gguf"
WORKING = Path("/kaggle/working")
MODEL_PATH = WORKING / GGUF_FILE
PORT = 8080
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are FinLens, an expert AI assistant specializing in Indian fintech, banking, and financial regulations. You provide accurate, well-structured answers about RBI regulations, SEBI guidelines, digital lending, UPI, NBFC compliance, PMLA, KYC requirements, and related topics.

When answering:
- Cite specific regulations, circulars, or guidelines when possible
- Mention the relevant authority (RBI, SEBI, FIU-IND, etc.)
- Provide actionable insights, not just definitions
- If you reference data or statistics, mention the source

You have access to a knowledge base of RBI circulars, fintech regulations, and financial compliance documents. Use this knowledge to provide grounded, accurate responses."""


def run(cmd, check=True):
    print(f"  → {cmd[:80]}...")
    return subprocess.run(cmd, shell=True, check=check)


def download_file(repo, filename, dest):
    run(f'python -c "'
        f"from huggingface_hub import hf_hub_download; "
        f"hf_hub_download(repo_id='{repo}', filename='{filename}', local_dir='{dest.parent}')"
        f'"')


def main():
    print("=" * 60)
    print("  FinLens Kaggle Deployment")
    print("=" * 60)

    # ── 1. Install cloudflared ──────────────────────────────────────────────
    print("\n[1/4] Installing cloudflared...")
    tunnel_bin = WORKING / "cloudflared"
    if not tunnel_bin.exists():
        run(f"wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O {tunnel_bin}")
        run(f"chmod +x {tunnel_bin}")

    # ── 2. Download model ──────────────────────────────────────────────────
    print("\n[2/4] Downloading GGUF model (~3GB, ~2min)...")
    if not MODEL_PATH.exists():
        download_file(GGUF_REPO, GGUF_FILE, MODEL_PATH)

    # ── 3. Install llama-cpp-python (pre-built wheel) ──────────────────────
    print("\n[3/4] Installing llama-cpp-python...")
    run("pip install -q llama-cpp-python[server] > /dev/null 2>&1")

    # Find the server binary
    server_bin = subprocess.run(
        "python -c \"import shutil; print(shutil.which('llama-server') or '')\"",
        shell=True, capture_output=True, text=True
    ).stdout.strip()

    if not server_bin:
        # Fallback: use llama_cpp module directly
        print("  Using llama-cpp-python server module...")
        server_cmd = f"python -m llama_cpp.server --model {MODEL_PATH} --host 0.0.0.0 --port {PORT} --n_ctx 8192 --n_threads 4"
    else:
        print(f"  Using: {server_bin}")
        server_cmd = f"{server_bin} --model {MODEL_PATH} --host 0.0.0.0 --port {PORT} --ctx-size 8192 --threads 4"

    # ── 4. Start server ────────────────────────────────────────────────────
    print("\n[4/4] Starting server...")

    # Kill any leftover process on the port
    run(f"lsof -ti:{PORT} | xargs -r kill -9 2>/dev/null || true")
    time.sleep(1)

    log_path = WORKING / "server.log"
    log_file = open(log_path, "w")
    server_proc = subprocess.Popen(
        server_cmd, shell=True,
        stdout=log_file, stderr=subprocess.STDOUT
    )

    # Wait for ready — model is 3GB, may take 30-60s to load
    print("  Waiting for server (model loading, ~30s)...")
    for i in range(120):
        time.sleep(3)
        # Check if process crashed
        if server_proc.poll() is not None:
            log_file.close()
            with open(log_path) as f:
                tail = f.read()[-1000:]
            print(f"  ERROR: Server exited\n{tail}")
            return
        # Try health endpoints
        try:
            import urllib.request
            # llama-cpp-python has /v1/models, not /health
            resp = urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=5)
            print(f"  Server ready! (took ~{(i+1)*3}s)")
            break
        except urllib.error.HTTPError as e:
            if e.code == 200:
                print(f"  Server ready! (took ~{(i+1)*3}s)")
                break
        except Exception:
            pass
    else:
        log_file.close()
        with open(log_path) as f:
            tail = f.read()[-1000:]
        print(f"  ERROR: Server timeout. Log:\n{tail}")
        return

    # ── 5. Start tunnel ────────────────────────────────────────────────────
    print("\n[5/5] Starting Cloudflare tunnel...")
    tunnel_proc = subprocess.Popen(
        [str(tunnel_bin), "tunnel", "--url", f"http://localhost:{PORT}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    public_url = None
    tunnel_log = WORKING / "tunnel.log"
    tl = open(tunnel_log, "w")
    for _ in range(60):
        time.sleep(2)
        line = tunnel_proc.stdout.readline().decode()
        tl.write(line)
        # Parse the hostname from the readiness line
        if "trycloudflare.com" in line:
            # e.g. "Your quick Tunnel has been created! Visit it at https://xxx.trycloudflare.com"
            import re
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if m:
                public_url = m.group(0)
                break
    tl.close()

    if not public_url:
        # Fallback: scan the whole log
        import re
        content = open(tunnel_log).read()
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", content)
        if m:
            public_url = m.group(0)
        else:
            print("  ERROR: Tunnel failed. Log:")
            print(content[-1500:])
            return

    # ── Done ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINLENS IS LIVE")
    print("=" * 60)
    print(f"\n  Public URL:  {public_url}")
    print(f"  Local URL:   http://localhost:{PORT}")
    print(f"\n  Test:")
    print(f'    curl -X POST {public_url}/v1/chat/completions \\')
    print(f'      -H "Content-Type: application/json" \\')
    print(f'      -d \'{{"messages":[{{"role":"user","content":"What is PMLA?"}}],"stream":true}}\'')
    print(f"\n  Frontend: set API_URL={public_url}")
    print(f"\n  Stop: !kill {server_proc.pid}")


if __name__ == "__main__":
    main()
