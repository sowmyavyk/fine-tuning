"""
FinLens Kaggle - Chat UI + Model
=================================
Full deployment: llama.cpp model server + Gradio chat UI + Cloudflare tunnel.
The PUBLIC URL serves a real chat page (no 404).

Setup:
1. New Kaggle notebook → GPU accelerator (T4)
2. Paste this full script → Run
3. Get public URL → share with client
"""

import os
import subprocess
import time
import re
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
GGUF_REPO = "Sowmyavyk/finlens-gguf"
GGUF_FILE = "finlens-merged.gguf"
WORKING = Path("/kaggle/working")
MODEL_PATH = WORKING / GGUF_FILE
API_PORT = 8080      # llama.cpp API
UI_PORT = 7860       # Gradio UI
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are FinLens, an expert AI assistant specializing in Indian fintech, banking, and financial regulations. You provide accurate, well-structured answers about RBI regulations, SEBI guidelines, digital lending, UPI, NBFC compliance, PMLA, KYC requirements, and related topics.

When answering:
- Cite specific regulations, circulars, or guidelines when possible
- Mention the relevant authority (RBI, SEBI, FIU-IND, etc.)
- Provide actionable insights, not just definitions
- If you reference data or statistics, mention the source

You have access to a knowledge base of RBI circulars, fintech regulations, and financial compliance documents. Use this knowledge to provide grounded, accurate responses."""


def run(cmd, check=True):
    print(f"  → {cmd[:70]}...")
    return subprocess.run(cmd, shell=True, check=check)


def wait_ready(url, port, timeout=180):
    import urllib.request, urllib.error
    for i in range(int(timeout / 3)):
        time.sleep(3)
        try:
            urllib.request.urlopen(url, timeout=5)
            return True
        except urllib.error.HTTPError:
            return True  # 404 means server is up
        except Exception:
            continue
    return False


def main():
    print("=" * 60)
    print("  FinLens Kaggle Deployment (Chat UI)")
    print("=" * 60)

    # ── 1. Cloudflared ─────────────────────────────────────────────────────
    print("\n[1/6] Installing cloudflared...")
    tunnel_bin = WORKING / "cloudflared"
    if not tunnel_bin.exists():
        run(f"wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O {tunnel_bin}")
        run(f"chmod +x {tunnel_bin}")

    # ── 2. Download model ──────────────────────────────────────────────────
    print("\n[2/6] Downloading GGUF model (~3GB, ~2min)...")
    if not MODEL_PATH.exists():
        run(f'python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id=\'{GGUF_REPO}\', filename=\'{GGUF_FILE}\', local_dir=\'{WORKING}\')"')

    # ── 3. Install deps ────────────────────────────────────────────────────
    print("\n[3/6] Installing server + UI...")
    run(f"pip install -q llama-cpp-python[server] gradio openai > /dev/null 2>&1")

    server_bin = subprocess.run(
        "python -c \"import shutil; print(shutil.which('llama-server') or '')\"",
        shell=True, capture_output=True, text=True
    ).stdout.strip()

    # ── 4. Start model API server ──────────────────────────────────────────
    print("\n[4/6] Starting model API server...")
    run(f"lsof -ti:{API_PORT} | xargs -r kill -9 2>/dev/null || true")
    time.sleep(1)

    if server_bin:
        server_cmd = f"{server_bin} --model {MODEL_PATH} --host 0.0.0.0 --port {API_PORT} --ctx-size 8192 --threads 4"
    else:
        server_cmd = f"python -m llama_cpp.server --model {MODEL_PATH} --host 0.0.0.0 --port {API_PORT} --n_ctx 8192 --n_threads 4"

    api_log = open(WORKING / "api.log", "w")
    api_proc = subprocess.Popen(server_cmd, shell=True, stdout=api_log, stderr=subprocess.STDOUT)

    if not wait_ready(f"http://localhost:{API_PORT}/v1/models", API_PORT):
        api_log.close()
        print("  ERROR: API server failed. Log:")
        print(open(WORKING / "api.log").read()[-1200:])
        return
    print(f"  Model API ready on :{API_PORT}")

    # ── 5. Start Gradio Chat UI ────────────────────────────────────────────
    print("\n[5/6] Starting Gradio chat UI...")
    run(f"lsof -ti:{UI_PORT} | xargs -r kill -9 2>/dev/null || true")
    time.sleep(1)

    ui_script = f'''
import gradio as gr
from openai import OpenAI

client = OpenAI(base_url="http://localhost:{API_PORT}/v1", api_key="not-needed")

SYSTEM = {SYSTEM_PROMPT!r}

def chat_fn(message, history):
    msgs = [{{"role": "system", "content": SYSTEM}}]
    for h in history or []:
        msgs.append({{"role": "user", "content": h[0]}})
        msgs.append({{"role": "assistant", "content": h[1]}})
    msgs.append({{"role": "user", "content": message}})
    stream = client.chat.completions.create(
        model="not-needed",
        messages=msgs,
        stream=True,
        max_tokens={MAX_TOKENS},
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta

with gr.Blocks(title="FinLens") as demo:
    gr.Markdown("# FinLens\\n\\nAI assistant for Indian fintech, banking & regulations. Ask about RBI, SEBI, PMLA, KYC, digital lending, UPI, NBFC compliance.")
    gr.ChatInterface(fn=chat_fn, type="messages").launch(server_name="0.0.0.0", server_port={UI_PORT})
'''

    ui_script_path = WORKING / "ui.py"
    ui_script_path.write_text(ui_script)
    ui_log = open(WORKING / "ui.log", "w")
    ui_proc = subprocess.Popen(
        f"python {ui_script_path}", shell=True,
        stdout=ui_log, stderr=subprocess.STDOUT
    )

    if not wait_ready(f"http://localhost:{UI_PORT}/", UI_PORT):
        ui_log.close()
        print("  ERROR: UI failed. Log:")
        print(open(WORKING / "ui.log").read()[-1200:])
        return
    print(f"  Chat UI ready on :{UI_PORT}")

    # ── 6. Start tunnel (auto-restarting watchdog -> chat UI) ───────────────
    print("\n[6/6] Starting Cloudflare tunnel (auto-restart watchdog) -> chat UI...")

    tunnel_bin = WORKING / "cloudflared"
    tunnel_cmd = [
        str(tunnel_bin), "tunnel", "--url", f"http://localhost:{UI_PORT}",
        "--protocol", "http2",   # http2 avoids the QUIC dead-lock issues
        "--no-autoupdate",
    ]

    url_file = WORKING / "public_url.txt"
    url_file.write_text("started...")

    url_pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    def extract_url(text):
        m = url_pattern.search(text)
        return m.group(0) if m else None

    # Watchdog thread: keeps cloudflared alive forever and captures the URL
    public_url_holder = {"url": None}

    def watchdog():
        proc = None
        import select as _select
        while True:
            try:
                if proc is None or proc.poll() is not None:
                    if proc is not None:
                        print("  ! tunnel died -> restarting")
                    proc = subprocess.Popen(
                        tunnel_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                r, _, _ = _select.select([proc.stdout], [], [], 2)
                if r:
                    chunk = os.read(proc.stdout.fileno(), 4096).decode(errors="replace")
                    for line in chunk.splitlines():
                        u = extract_url(line)
                        if u and u != public_url_holder["url"]:
                            public_url_holder["url"] = u
                            url_file.write_text(u)
                            print(f"  >>> PUBLIC URL: {u}   (saved to /kaggle/working/public_url.txt)", flush=True)
                        elif url_pattern.search(line):
                            u2 = extract_url(line)
                            if u2 and u2 != public_url_holder["url"]:
                                public_url_holder["url"] = u2
                                url_file.write_text(u2)
                                print(f"  >>> PUBLIC URL: {u2}   (saved to /kaggle/working/public_url.txt)", flush=True)
            except Exception as e:
                print(f"  watchdog err: {e}")
                time.sleep(3)

    import threading
    t = threading.Thread(target=watchdog, daemon=True)
    t.start()

    # Wait for first URL (up to 5 min), then show the LIVE banner
    print("  Waiting for tunnel URL (usually ~30s)...")
    for _ in range(150):
        if public_url_holder["url"]:
            break
        time.sleep(2)

    public_url = public_url_holder["url"]

    print("\n" + "=" * 60)
    if public_url:
        print("  FINLENS IS LIVE — open in browser:")
        print(f"\n    {public_url}\n")
        print("  This URL serves the actual chat interface.")
        print(f"  Current URL always saved at /kaggle/working/public_url.txt")
    else:
        print("  Tunnel URL not captured yet — SEE RUNNING OUTPUT above/below.")
        print("  It auto-saves to /kaggle/working/public_url.txt once assigned.")
    print("  Model API:  http://localhost:" + str(API_PORT) + "/v1")
    print("  Stop: !kill " + str(api_proc.pid) + " " + str(ui_proc.pid))
    print("=" * 60)
    print("\n  Cell stays running (tunnel auto-restarts if it drops).")
    print("  To refresh the URL, read:  !cat /kaggle/working/public_url.txt")

    # Keep the cell alive while the watchdog thread runs
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
