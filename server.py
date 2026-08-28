import asyncio
import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from citation_lookup import build_index
from chatbot import SYSTEM_PROMPT, ADAPTER_DIR, BASE_MODEL

app = FastAPI(title="FinLens Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
TOKENIZER = None
USING_ADAPTER = None
SOURCE_INDEX = None


class ChatRequest(BaseModel):
    message: str
    history: list = []
    temperature: float = 0.4
    max_tokens: int = 1024
    num_citations: int = 3


def load_model():
    global MODEL, TOKENIZER, USING_ADAPTER
    from mlx_lm import load

    adapter_path = str(ADAPTER_DIR) if ADAPTER_DIR.exists() else None
    MODEL, TOKENIZER = load(BASE_MODEL, adapter_path=adapter_path)
    USING_ADAPTER = adapter_path is not None
    print(f"[server] model loaded. adapter: {USING_ADAPTER}")


def retrieve_context(message: str, k: int = 3):
    global SOURCE_INDEX
    if SOURCE_INDEX is None:
        SOURCE_INDEX = build_index()
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
        "model": BASE_MODEL,
        "using_adapter": USING_ADAPTER,
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

    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    text = TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    async def event_stream():
        from mlx_lm import generate as gen
        from mlx_lm.sample_utils import make_sampler as ms
        from mlx_lm.sample_utils import make_repetition_penalty as mrp
        from clean_stream import clean_stream

        sampler = ms(temp=req.temperature, top_p=0.9)
        stream = gen(
            MODEL,
            TOKENIZER,
            prompt=text,
            max_tokens=req.max_tokens,
            sampler=sampler,
            logits_processors=[mrp(1.15)],
            verbose=False,
        )
        for piece in clean_stream(stream):
            yield f"data: {json.dumps({'type': 'token', 'content': piece})}\n\n"
            await asyncio.sleep(0)
        yield f"data: {json.dumps({'type': 'sources', 'sources': citations})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
