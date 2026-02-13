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

RUN mkdir -p /app/models/embeddings /app/models/cross-encoder

# Create a Python script for downloading models
RUN echo 'import os\n\
import sys\n\
from google.cloud import storage\n\
\n\
if os.environ.get("CLOUD_RUN_ENV") == "True" and os.environ.get("BUCKET_NAME"):\n\
    print("Downloading models from GCS...", flush=True)\n\
    bucket_name = os.environ["BUCKET_NAME"]\n\
    client = storage.Client()\n\
    bucket = client.bucket(bucket_name)\n\
    \n\
    # Download main model\n\
    model_path = "/app/models/legal_qwen.Q4_K_M.gguf"\n\
    if not os.path.exists(model_path):\n\
        print("Downloading main model...", flush=True)\n\
        blob = bucket.blob("legal_qwen.Q4_K_M.gguf")\n\
        blob.download_to_filename(model_path)\n\
        print("✓ Model downloaded", flush=True)\n\
    \n\
    # Download embeddings\n\
    print("Downloading embeddings...", flush=True)\n\
    for blob in bucket.list_blobs(prefix="embeddings/"):\n\
        if not blob.name.endswith("/"):\n\
            local_path = f"/app/models/{blob.name}"\n\
            os.makedirs(os.path.dirname(local_path), exist_ok=True)\n\
            if not os.path.exists(local_path):\n\
                blob.download_to_filename(local_path)\n\
                print(f"✓ {blob.name}", flush=True)\n\
    \n\
    # Download cross-encoder\n\
    print("Downloading cross-encoder...", flush=True)\n\
    for blob in bucket.list_blobs(prefix="cross-encoder/"):\n\
        if not blob.name.endswith("/"):\n\
            local_path = f"/app/models/{blob.name}"\n\
            os.makedirs(os.path.dirname(local_path), exist_ok=True)\n\
            if not os.path.exists(local_path):\n\
                blob.download_to_filename(local_path)\n\
                print(f"✓ {blob.name}", flush=True)\n\
    \n\
    print("✓ All downloads complete", flush=True)\n\
else:\n\
    print("Skipping model download (not in Cloud Run)", flush=True)\n\
' > /app/download_models.py

EXPOSE 8080

# Download models then start Streamlit
CMD python /app/download_models.py && \
    streamlit run /app/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0