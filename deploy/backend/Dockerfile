FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve.py citation_lookup.py clean_stream.py ./
COPY data ./data
# The HF adapter (PEFT) ships with the image so no runtime download is needed.
COPY model/adapter ./model/adapter

# 8000 = FastAPI; 7860 left free for Gradio variants
EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]