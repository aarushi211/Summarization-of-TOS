# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# pip installs console_scripts into ~/.local/bin — put it on PATH during build (silences warnings)
ENV PATH=/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake git \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# 1. Install CPU-only Torch first (Crucial for size reduction)
RUN pip install --user --no-cache-dir \
    torch==2.2.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 2. Install production requirements
COPY api/requirements.prod.txt .
RUN pip install --user --no-cache-dir -r requirements.prod.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgomp1 \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# ── Application code ──────────────────────────────────────────────────────────
COPY api/ ./api/
COPY src/ ./src/

# ── Models ────────────────────────────────────────────────────────────────────
# OPTION A: Bake models into image (Image size will be >3GB)
# UNCOMMENT these lines if you want a self-contained image for Cloud Run
# COPY models/legal_qwen.Q4_K_M.gguf ./models/legal_qwen.Q4_K_M.gguf
# COPY models/embeddings/              ./models/embeddings/
# COPY models/cross-encoder/           ./models/cross-encoder/

# OPTION B: Mount models at runtime (Image size will be ~500MB)
# Recommended for local testing and Cloud Run with Volume Mounts.

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/legal_qwen.Q4_K_M.gguf \
    LANGCHAIN_TRACING_V2=true \
    LANGCHAIN_PROJECT=tos-summarizer-prod

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# CMD ["uvicorn", "api.main:app", \
#      "--host", "0.0.0.0", \
#      "--port", "8080", \
#      "--workers", "1", \
#      "--timeout-keep-alive", "75", \
#      "--log-level", "warning"]
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --timeout-keep-alive 75 --log-level warning"]
