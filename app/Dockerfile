FROM python:3.10-slim

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

RUN apt-get purge -y --auto-remove build-essential

COPY src/RAG/ /app/src/RAG/
COPY app/app.py /app/app.py
COPY app/download_models.py /app/download_models.py

RUN mkdir -p /app/models/embeddings /app/models/cross-encoder

EXPOSE 8080

CMD python /app/download_models.py && \
    streamlit run /app/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0