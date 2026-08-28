"""FinLens inference backend — portability layer.

Runs the HF (transformers + PEFT) stack so the same app can be deployed on
any Linux VPS / Docker host / NVIDIA or Apple-Silicon GPU, unlike the Mac-only
MLX path. Exposes the identical SSE /api/chat contract as server.py so the
Next.js frontend works unchanged.

Env vars:
  MODEL_NAME    base model HF id or path         (default Qwen/Qwen2.5-1.5B)
  ADAPTER_DIR   PEFT LoRA adapter dir            (default ./model/adapter)
  DATA_DIR      dir holding fintech_data_grounded*.json
  MAX_TOKENS    max generated tokens             (default 1024)
  TEMP          sampling temperature             (default 0.4)
  DEVICE        auto | cpu | cuda | mps          (default auto)
"""

import asyncio
import json
import os
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from citation_lookup import build_index
from clean_stream import clean_stream

BASE = Path(__file__).parent
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B")
ADAPTER_DIR = Path(os.getenv("ADAPTER_DIR", str(BASE / "model" / "adapter")))
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE)))
DEVICE = os.getenv("DEVICE", "auto")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMP = float(os.getenv("TEMP", "0.4"))

SYSTEM_PROMPT = (
    "You are an Indian fintech regulatory compliance expert grounded in RBI, SEBI, "
    "FIU-IND, and PMLA regulations. Give DETAILED, comprehensive answers: define the "
    "terms, explain the purpose, cite the relevant acts/sections (PMLA 2002, RBI KYC "
    "Master Directions, FIU-IND reporting rules), list practical obligations and "
    "step-by-step processes, and include examples. Write in full paragraphs — do not "
    "be brief. If you are not sure, say so rather than guessing."
)

app = FastAPI(title="FinLens Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
TOKENIZER = None
SOURCE_INDEX = None


class ChatRequest(BaseModel):
    message: str
    history: list = []
    temperature: float = TEMP
    max_tokens: int = MAX_TOKENS
    num_citations: int = 3


def _grounded_files():
    return [
        p for p in (DATA_DIR / "data").glob("fintech_data_grounded*.json")
    ] if (DATA_DIR / "data").is_dir() else [p for p in DATA_DIR.glob("fintech_data_grounded*.json")]


def load_model():
    global MODEL, TOKENIZER
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_map = None
    torch_dtype = torch.float32
    if DEVICE == "cuda":
        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device_map == "cuda:0" else torch.float32
    elif DEVICE == "mps":
        device_map = "mps"
        torch_dtype = torch.float16
    elif DEVICE == "cpu":
        device_map = "cpu"
    else:  # auto
        if torch.cuda.is_available():
            device_map, torch_dtype = "cuda:0", torch.float16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device_map, torch_dtype = "mps", torch.float16

    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map=device_map,
    ).eval()
    use_adapter = ADAPTER_DIR.exists()
    if use_adapter:
        MODEL = PeftModel.from_pretrained(MODEL, ADAPTER_DIR)
        MODEL.eval()
    print(
        f"[serve] model {MODEL_NAME} loaded on {device_map or 'cpu'} "
        f"(adapter={'yes' if use_adapter else 'NO'})"
    )


def retrieve_context(message: str, k: int = 3):
    global SOURCE_INDEX
    if SOURCE_INDEX is None:
        files = _grounded_files()
        SOURCE_INDEX = build_index(files)
    hits = SOURCE_INDEX.search(message, k=k)
    if not hits:
        return "", []
    context = "Relevant source material (use it to ground your answer):\n"
    citations = []
    for i, item in enumerate(hits, 1):
        p = item["pair"]
        context += (
            f"[{i}] Q: {p.get('instruction', '')}\n"
            f"    A: {p.get('response', '')}\n\n"
        )
        citations.append(SOURCE_INDEX.citation(item))
    return context.strip(), citations


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "using_adapter": ADAPTER_DIR.exists(),
        "pairs_indexed": len(SOURCE_INDEX.pairs) if SOURCE_INDEX else 0,
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    if MODEL is None:
        load_model()

    context, citations = retrieve_context(req.message, req.num_citations)
    user_content = f"{context}\n\nQuestion: {req.message}" if context else req.message

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for entry in req.history:
        role = entry.get("role", "user") if isinstance(entry, dict) else "user"
        content = entry.get("content", str(entry)) if isinstance(entry, dict) else str(entry)
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_content})

    text = TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = TOKENIZER(text, return_tensors="pt")
    if hasattr(MODEL, "device") and str(MODEL.device) != "cpu":
        inputs = inputs.to(MODEL.device)

    stream = token_stream(
        inputs.input_ids,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )

    async def event_stream():
        for piece in stream:
            yield f"data: {json.dumps({'type': 'token', 'content': piece})}\n\n"
            await asyncio.sleep(0)
        yield f"data: {json.dumps({'type': 'sources', 'sources': citations})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def token_stream(input_ids, max_tokens: int = MAX_TOKENS, temperature: float = TEMP):
    """Yield decoded text pieces with repetition penalty + clean_stream guards."""
    import torch
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(TOKENIZER, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.15,
        eos_token_id=TOKENIZER.eos_token_id,
        pad_token_id=TOKENIZER.pad_token_id,
        streamer=streamer,
    )
    thread = threading.Thread(
        target=MODEL.generate, kwargs=generation_kwargs, daemon=True
    )
    thread.start()
    return clean_stream((p for p in streamer))


if __name__ == "__main__":
    import uvicorn

    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)