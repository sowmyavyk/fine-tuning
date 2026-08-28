"""
FinLens Kaggle Deployment
=========================
Run this script in a Kaggle notebook (GPU accelerator: T4 or P100).
It downloads the GGUF model + RAG data from HF Hub, starts llama.cpp server,
and exposes it via a Cloudflare tunnel.

Steps:
1. Create a new Kaggle notebook
2. Enable GPU accelerator (Settings → Accelerator → GPU T4 ×2 or P100)
3. Add a new cell and paste this entire script
4. Run the cell — the public URL will be printed at the end
5. Use the URL with the FinLens frontend or test with curl

Free tier: 30h GPU/week, unlimited CPU. No card needed.
"""

import os
import subprocess
import time
import sys
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
GGUF_REPO = "Sowmyavyk/finlens-gguf"
GGUF_FILE = "finlens-merged.gguf"
DATA_DIR = Path("/kaggle/working/finlens-data")
MODEL_DIR = Path("/kaggle/working/finlens-model")
LLAMA_SERVER = Path("/kaggle/working/llama-server")
PORT = 8080
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are FinLens, an expert AI assistant specializing in Indian fintech, banking, and financial regulations. You provide accurate, well-structured answers about RBI regulations, SEBI guidelines, digital lending, UPI, NBFC compliance, PMLA, KYC requirements, and related topics.

When answering:
- Cite specific regulations, circulars, or guidelines when possible
- Mention the relevant authority (RBI, SEBI, FIU-IND, etc.)
- Provide actionable insights, not just definitions
- If you reference data or statistics, mention the source

You have access to a knowledge base of RBI circulars, fintech regulations, and financial compliance documents. Use this knowledge to provide grounded, accurate responses."""


def run(cmd, check=True, **kwargs):
    print(f"  → {cmd}")
    return subprocess.run(cmd, shell=True, check=check, **kwargs)


def install_dependencies():
    print("\n[1/6] Installing dependencies...")
    run("apt-get update -qq && apt-get install -y -qq curl wget cmake build-essential > /dev/null 2>&1")
    run("pip install -q huggingface_hub > /dev/null 2>&1")


def download_model():
    print("\n[2/6] Downloading GGUF model from HF Hub...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    run(f'python -c "'
        f"from huggingface_hub import hf_hub_download; "
        f"hf_hub_download(repo_id='{GGUF_REPO}', filename='{GGUF_FILE}', "
        f"local_dir='{MODEL_DIR}')"
        f'"')
    print(f"  Model: {MODEL_DIR / GGUF_FILE}")


def download_rag_data():
    print("\n[3/6] Downloading RAG data...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ["fintech_data_grounded.json", "fintech_data_grounded_rbi.json"]:
        run(f'python -c "'
            f"from huggingface_hub import hf_hub_download; "
            f"hf_hub_download(repo_id='{GGUF_REPO}', filename='data/{fname}', "
            f"local_dir='{MODEL_DIR}')"
            f'"')
    # Move data files to DATA_DIR
    for fname in ["fintech_data_grounded.json", "fintech_data_grounded_rbi.json"]:
        src = MODEL_DIR / "data" / fname
        dst = DATA_DIR / fname
        if src.exists():
            src.rename(dst)
    print(f"  RAG data: {DATA_DIR}")


def build_llama_cpp():
    print("\n[4/6] Building llama.cpp server...")
    if LLAMA_SERVER.exists():
        print("  Already built, skipping.")
        return

    run("git clone --depth 1 https://github.com/ggml-org/llama.cpp /kaggle/working/llama.cpp-src > /dev/null 2>&1")
    run("cmake -B /kaggle/working/llama.cpp-src/build "
        "-S /kaggle/working/llama.cpp-src "
        "-DLLAMA_CURL=OFF "
        "-DBUILD_SHARED_LIBS=OFF "
        "-DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1",
        cwd="/kaggle/working/llama.cpp-src")
    run("cmake --build /kaggle/working/llama.cpp-src/build --target llama-server -j$(nproc) > /dev/null 2>&1",
        cwd="/kaggle/working/llama.cpp-src")

    # Copy the built binary
    import shutil
    src_bin = Path("/kaggle/working/llama.cpp-src/build/bin/llama-server")
    if not src_bin.exists():
        # Try alternative path
        src_bin = Path("/kaggle/working/llama.cpp-src/build/bin/llama-server")
        if not src_bin.exists():
            raise FileNotFoundError("llama-server binary not found after build")

    shutil.copy2(src_bin, LLAMA_SERVER)
    print(f"  Built: {LLAMA_SERVER}")


def start_server():
    print("\n[5/6] Starting llama.cpp server...")
    model_path = MODEL_DIR / GGUF_FILE

    cmd = [
        str(LLAMA_SERVER),
        "--model", str(model_path),
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--n-predict", str(MAX_TOKENS),
        "--ctx-size", "8192",
        "--threads", "4",
        "--parallel", "2",
    ]

    # Start server in background
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    print("  Waiting for server to start...")
    for i in range(60):
        time.sleep(2)
        try:
            import urllib.request
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
            print(f"  Server ready on port {PORT}")
            return proc
        except Exception:
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode()
                raise RuntimeError(f"Server exited: {stderr}")
            continue

    raise RuntimeError("Server failed to start within 120s")


def setup_tunnel():
    print("\n[6/6] Setting up Cloudflare tunnel...")
    tunnel_bin = Path("/kaggle/working/cloudflared")

    if not tunnel_bin.exists():
        run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
            f"-O {tunnel_bin}")
        run(f"chmod +x {tunnel_bin}")

    # Start tunnel in background
    proc = subprocess.Popen(
        [str(tunnel_bin), "tunnel", "--url", f"http://localhost:{PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for tunnel URL
    print("  Waiting for tunnel URL...")
    public_url = None
    for i in range(60):
        time.sleep(2)
        line = proc.stdout.readline().decode()
        if "trycloudflare.com" in line:
            # Extract URL
            for word in line.split():
                if "trycloudflare.com" in word:
                    public_url = word.strip()
                    break
        if public_url:
            break

    if not public_url:
        raise RuntimeError("Cloudflare tunnel failed to start")

    return public_url


def main():
    print("=" * 60)
    print("  FinLens Kaggle Deployment")
    print("  Free GPU: 30h/week (T4/P100)")
    print("=" * 60)

    install_dependencies()
    download_model()
    download_rag_data()
    build_llama_cpp()
    server_proc = start_server()
    public_url = setup_tunnel()

    print("\n" + "=" * 60)
    print("  DEPLOYMENT READY")
    print("=" * 60)
    print(f"\n  Public URL: {public_url}")
    print(f"  Local URL:  http://localhost:{PORT}")
    print(f"\n  Test with curl:")
    print(f'    curl -X POST {public_url}/v1/chat/completions \\')
    print(f'      -H "Content-Type: application/json" \\')
    print(f'      -d \'{{"messages": [{{"role": "user", "content": "What is PMLA?"}}], "stream": true}}\'')
    print(f"\n  Or connect the FinLens frontend:")
    print(f"    Set API_URL={public_url}")
    print(f"\n  To stop: kill the server process")
    print(f"    !kill {server_proc.pid}")


if __name__ == "__main__":
    main()
