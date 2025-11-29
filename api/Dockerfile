FROM python:3.10-slim

# Work dir
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY api/ .

# Copy model artifact
COPY data/model_artifacts/led_onnx_quantized/ ./model_artifacts/led_onnx_quantized/

# Expose Port
EXPOSE 8080

Run Command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]