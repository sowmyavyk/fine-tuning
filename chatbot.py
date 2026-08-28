import json
import os
from pathlib import Path

import gradio as gr

from citation_lookup import build_index

BASE_MODEL = "mlx-community/Qwen2.5-1.5B-4bit"
ADAPTER_DIR = Path("fintech_finetuned_qwen/adapters_colab")

SYSTEM_PROMPT = (
    "You are an Indian fintech regulatory compliance expert grounded in RBI, SEBI, "
    "FIU-IND, and PMLA regulations. Give DETAILED, comprehensive answers: define the "
    "terms, explain the purpose, cite the relevant acts/sections (PMLA 2002, RBI KYC "
    "Master Directions, FIU-IND reporting rules), list practical obligations and "
    "step-by-step processes, and include examples. Write in full paragraphs — do not "
    "be brief. If you are not sure, say so rather than guessing."
)

MODEL = None
TOKENIZER = None
USING_ADAPTER = False
SOURCE_INDEX = None


def load_model():
    global MODEL, TOKENIZER, USING_ADAPTER
    from mlx_lm import load

    adapter_path = str(ADAPTER_DIR) if ADAPTER_DIR.exists() else None
    MODEL, TOKENIZER = load(BASE_MODEL, adapter_path=adapter_path)
    USING_ADAPTER = adapter_path is not None
    return USING_ADAPTER


def retrieve_context(message: str, k: int = 3) -> tuple:
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


def respond(message, history):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
    from clean_stream import clean_stream

    if MODEL is None:
        load_model()

    context, citations = retrieve_context(message)
    user_content = message
    if context:
        user_content = f"{context}\n\nQuestion: {message}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for entry in history:
        if isinstance(entry, dict):
            role = entry.get("role", "user")
            content = entry.get("content", "")
        else:
            role, content = "user", str(entry)
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_content})

    text = TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    stream = generate(
        MODEL,
        TOKENIZER,
        prompt=text,
        max_tokens=1024,
        sampler=make_sampler(temp=0.4, top_p=0.9),
        logits_processors=[make_repetition_penalty(1.15)],
        verbose=False,
    )
    for piece in clean_stream(stream):
        yield piece

    if citations:
        yield "\n\n---\n**Sources:**\n" + "\n".join(
            f"- [{c['source_doc']}]({c['url']})" for c in citations
        )


adapter_used = load_model()
title = "FinLens Chatbot 🪙"
if adapter_used:
    title += " (fine-tuned adapter loaded)"
else:
    title += " (base model — run training to load adapter)"

demo = gr.ChatInterface(
    fn=respond,
    title=title,
    description=(
        "Ask about CKYC, KYC, AML, VKYC, CERSAI, PMLA 2002, RBI master directions, "
        "FIU-IND obligations, and more."
        + ("" if adapter_used else "\n\n⚠️ No fine-tuned adapter found yet — using the base model.")
    ),
    examples=[
        "What is CKYC and how is it different from regular KYC?",
        "What are the reporting obligations for AML under PMLA 2002?",
        "What is VKYC vs e-KYC?",
        "What is the process for CERSAI registration?",
    ],
)

if __name__ == "__main__":
    demo.launch()
