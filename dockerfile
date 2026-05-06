# Use python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Copy requirements.txt from your local machine to the container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
# We copy the entire 'app' and 'src' folders so imports work
COPY app/ app/
COPY src/RAG/ src/RAG/

# Create the directory for the model
RUN mkdir -p data/model_artifacts/

# ---------------------------------------------------------------------------
# MODEL DOWNLOAD STEP
# Hugging Face Spaces cannot handle large Git LFS pushes easily.
# We download the model *inside* the container build process.
# ---------------------------------------------------------------------------
RUN pip install huggingface-hub

# Replace 'your-username/mistral-tos-quantized' with the Repo ID where you uploaded the GGUF file
# Replace 'mistral_tos.Q4_K_M.gguf' with the exact filename
RUN huggingface-cli download aarushi-211/mistral-tos-quantized mistral_tos.Q4_K_M.gguf \
    --local-dir data/model_artifacts/ \
    --local-dir-use-symlinks False

# Expose port 7860 (Standard for HF Spaces)
EXPOSE 7860

# Run Streamlit
# We point to app/app.py
CMD ["streamlit", "run", "app/app.py", "--server.port=7860", "--server.address=0.0.0.0"]