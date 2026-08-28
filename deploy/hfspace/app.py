"""FinLens HF Space - ZeroGPU Gradio chat app.

Loads Qwen2.5-1.5B + LoRA adapter (bf16) on ZeroGPU and serves a
streaming Gradio chat UI grounded in the RBI/SEBI/FIU-IND/PMLA Q&A pairs, with
source citations appended to each answer.

ZeroGPU rules honored:
  - `import spaces` precedes every CUDA-touching import.
  - Model loads on CPU at module scope (PEFT-safe); the single .to("cuda")
    afterwards is intercepted and packed by the ZeroGPU backend.
  - The function Gradio binds is the one decorated with @spaces.GPU.
  - @spaces.GPU supports generators, so token streaming works via yield.
  - Read-only globals only; no CUDA tensors cross the worker boundary.

Env vars:
  MODEL_ID         base model repo   (default Qwen/Qwen2.5-1.5B)
  ADAPTER_ID       LoRA adapter repo (default Sowmyavyk/qwen-fintech-adapter)
  MAX_TOKENS       max new tokens    (default 1024)
  TEMP             sampling temp     (default 0.4)
  NUM_CITATIONS    RAG hits          (default 3)
  HF_TOKEN         secret with read access to the private adapter
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import spaces  # noqa: E402  must precede torch
import torch  # noqa: E402

import gradio as gr  # noqa: E402

from pathlib import Path  # noqa: E402
from threading import Thread  # noqa: E402

from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from peft import PeftModel  # noqa: E402

from citation_lookup import build_index  # noqa: E402
from clean_stream import clean_stream  # noqa: E402

BASE = Path(__file__).parent
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B")
ADAPTER_ID = os.getenv("ADAPTER_ID", "Sowmyavyk/qwen-fintech-adapter")
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE / "data")))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMP = float(os.getenv("TEMP", "0.4"))
NUM_CITATIONS = int(os.getenv("NUM_CITATIONS", "3"))

SYSTEM_PROMPT = (
    "You are an Indian fintech regulatory compliance expert grounded in RBI, SEBI, "
    "FIU-IND, and PMLA regulations. Give DETAILED, comprehensive answers: define the "
    "terms, explain the purpose, cite the relevant acts/sections (PMLA 2002, RBI KYC "
    "Master Directions, FIU-IND reporting rules), list practical obligations and "
    "step-by-step processes, and include examples. Write in full paragraphs - do not "
    "be brief. If you are not sure, say so rather than guessing."
)

# ---------------------------------------------------------------------------
# Module-scope model load. On ZeroGPU, `import spaces` intercepts .to("cuda")
# at module scope and the backend "packs" the weights to disk at startup; the
# worker streams them into VRAM on first use.
#
# The base model is loaded on CPU (bf16) and the LoRA adapter is applied while
# still on CPU - PEFT's safetensors loader would otherwise pick `cuda` via
# infer_device() (spaces patches torch.cuda.is_available() to True in the main
# process, where no real GPU exists) and fail. `torch_device="cpu"` pins the
# adapter load to CPU. Only after the adapter is attached do we move everything
# with one .to("cuda"), which the ZeroGPU patch intercepts and packs.
# ZeroGPU's 48GB VRAM needs no 4-bit.
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

model = PeftModel.from_pretrained(
    model, ADAPTER_ID, torch_device="cpu", autocast_adapter_dtype=False
).eval()
print(f"[space] adapter {ADAPTER_ID} loaded (cpu)")

model = model.to("cuda")
print(f"[space] model {MODEL_ID} loaded (bf16, cuda)")

# Read-only RAG index, built once at module scope.
SOURCE_INDEX = build_index(list(DATA_DIR.glob("fintech_data_grounded*.json")))


def retrieve_context(message: str, k: int = NUM_CITATIONS):
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


def _estimate_duration(message: str, history, *args, **kwargs):
    # Reserve a realistic GPU budget instead of a fixed generous value.
    # ZeroGPU charges the RESERVED duration against the visitor's daily quota
    # (300s free / 2400s PRO), compared as requested-vs-remaining, so declaring
    # only what we need both wastes less quota and ranks higher in the queue.
    # ~1024 tokens at a conservative ~20 tok/s on the shared Blackwell = ~50s;
    # cap at 120s so a long answer never gets truncated.
    worst = max(20, int(MAX_TOKENS / 20))
    return min(120, worst)


@spaces.GPU(duration=_estimate_duration)
def chat(message: str, history):
    context, citations = retrieve_context(message)
    user_content = f"{context}\n\nQuestion: {message}" if context else message

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for entry in history:
        if isinstance(entry, dict):
            role = entry.get("role", "user")
            content = entry.get("content", str(entry))
        else:
            role, content = "user", str(entry)
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_content})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = dict(
        input_ids=inputs["input_ids"],
        max_new_tokens=MAX_TOKENS,
        do_sample=True,
        temperature=TEMP,
        top_p=0.9,
        repetition_penalty=1.15,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
    )
    Thread(target=model.generate, kwargs=generation_kwargs, daemon=True).start()

    answer = ""
    for piece in clean_stream(iter(streamer)):
        answer += piece
        yield answer

    if citations:
        lines = [f"- **{c['source_doc']}** - {c['url']}" for c in citations]
        answer += "\n\n### Sources\n" + "\n".join(lines)
    yield answer


demo = gr.ChatInterface(
    fn=chat,
    title="FinLens - Indian Fintech Compliance Copilot",
    description=(
        "Qwen2.5-1.5B fine-tuned on RBI / SEBI / FIU-IND / PMLA material and "
        "grounded in 23,000+ curated Q&A pairs. Answers include source citations."
    ),
    examples=[
        "What is VKYC?",
        "Explain the reporting obligations under PMLA for a payment aggregator.",
        "What KYC norms apply under the RBI Master Directions?",
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)