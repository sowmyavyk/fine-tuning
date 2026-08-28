---
title: FinLens Chat
emoji: 📋
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
python_version: "3.12"
pinned: false
---

# FinLens Chat

Indian fintech compliance copilot: Qwen2.5-1.5B + LoRA adapter (4-bit via
bitsandbytes), RAG-grounded with RBI / PMLA citations. Streaming Gradio chat UI.

Runs on **ZeroGPU** (free): the GPU is allocated per request, so the model is
loaded at module scope and the Gradio event handler is decorated with
`@spaces.GPU`. Free personal accounts can host up to 2 ZeroGPU Spaces.

- Model: `Qwen/Qwen2.5-1.5B` (not gated, Apache-2.0)
- Adapter: `Sowmyavyk/qwen-fintech-adapter` (private - fetched at runtime via the
  `HF_TOKEN` secret, so the LoRA weights never ship in this public repo)
- Data: 23,000+ grounded Q&A pairs in `data/` (bundled here)