FROM python:3.10-slim

WORKDIR /app

# Install minimal system deps
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# 1. Install llama-cpp-python BINARY explicitly
# We use --extra-index-url AND --prefer-binary to ensure it grabs the pre-built wheel
RUN pip install llama-cpp-python --prefer-binary

# 2. Install other Python deps
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY api/ .

# Copy GGUF Model
COPY models/legal_qwen.Q4_K_M.gguf ./models/legal_qwen.Q4_K_M.gguf

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]