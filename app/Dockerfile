FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')"
# COPY models/embeddings/ /root/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/
COPY app/ app/
COPY src/RAG/ src/RAG/

RUN mkdir -p /app/models/embeddings

COPY models/legal_qwen.Q4_K_M.gguf /app/models/legal_qwen.Q4_K_M.gguf
COPY models/embeddings/ /app/models/embeddings/


EXPOSE 8080

CMD ["streamlit", "run", "app/app.py", "--server.port=8080", "--server.address=0.0.0.0"]