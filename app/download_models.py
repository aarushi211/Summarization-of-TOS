import os
from google.cloud import storage

if os.environ.get("CLOUD_RUN_ENV") == "True" and os.environ.get("BUCKET_NAME"):
    print("Downloading models from GCS...", flush=True)
    bucket_name = os.environ["BUCKET_NAME"]
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    model_path = "/app/models/legal_qwen.Q4_K_M.gguf"
    if not os.path.exists(model_path):
        print("Downloading main model...", flush=True)
        blob = bucket.blob("legal_qwen.Q4_K_M.gguf")
        blob.download_to_filename(model_path)
        print("✓ Model downloaded", flush=True)
    
    print("Downloading embeddings...", flush=True)
    for blob in bucket.list_blobs(prefix="embeddings/"):
        if not blob.name.endswith("/"):
            local_path = f"/app/models/{blob.name}"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if not os.path.exists(local_path):
                blob.download_to_filename(local_path)
                print(f"✓ {blob.name}", flush=True)
    
    print("Downloading cross-encoder...", flush=True)
    for blob in bucket.list_blobs(prefix="cross-encoder/"):
        if not blob.name.endswith("/"):
            local_path = f"/app/models/{blob.name}"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if not os.path.exists(local_path):
                blob.download_to_filename(local_path)
                print(f"✓ {blob.name}", flush=True)
    
    print("✓ All downloads complete", flush=True)
else:
    print("Skipping model download (not in Cloud Run)", flush=True)