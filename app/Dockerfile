FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/RAG/ /app/src/RAG/
COPY app/app.py /app/app.py
# COPY models/legal_qwen.Q4_K_M.gguf app/models/legal_qwen.Q4_K_M.gguf
# COPY models/cross-encoder/ms-marco-MiniLM-L-6-v2 app/models/cross-encoder/ms-marco-MiniLM-L-6-v2
# COPY models/embeddings app/models/embeddings

EXPOSE 8080

CMD ["sh", "-c", "streamlit run /app/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]