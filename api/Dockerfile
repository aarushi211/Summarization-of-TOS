# ── Stage 1: Builder ─────────────────────────────────────────────────────────
# Compiles all native extensions (llama-cpp, tokenizers) in an isolated layer.
# This layer is never shipped — only the compiled artifacts are copied forward.
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake git \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install llama-cpp-python first (heaviest native dep — separate cache layer)
RUN pip install --user --no-cache-dir \
    "llama-cpp-python==0.2.90" \
    --prefer-binary

# Install everything else from the pinned lockfile
COPY api/requirements.lock.txt .
RUN pip install --user --no-cache-dir -r requirements.lock.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
# Minimal image — only what's needed to run the application.
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system libraries only (no compilers)
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
# Copy only what the API needs — not the entire repo
COPY api/main.py            ./api/main.py
COPY src/RAG/rag_pipeline.py ./src/RAG/rag_pipeline.py
COPY src/__init__.py         ./src/__init__.py
COPY src/RAG/__init__.py     ./src/RAG/__init__.py

# ── Model ─────────────────────────────────────────────────────────────────────
# The GGUF model is baked into the image for simplicity.
# For faster deploys without model changes, consider mounting via Cloud Run
# Volume Mounts from a GCS bucket instead and removing this line.
COPY models/legal_qwen.Q4_K_M.gguf ./models/legal_qwen.Q4_K_M.gguf

# Embedding and cross-encoder models are loaded from the path set in the env.
# They should also be baked in or mounted the same way.
COPY models/embeddings/              ./models/embeddings/
COPY models/cross-encoder/           ./models/cross-encoder/

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/legal_qwen.Q4_K_M.gguf \
    # LangSmith tracing — set LANGCHAIN_API_KEY via Secret Manager at runtime
    LANGCHAIN_TRACING_V2=true \
    LANGCHAIN_PROJECT=tos-summarizer-prod

EXPOSE 8080

# ── Health check ──────────────────────────────────────────────────────────────
# Cloud Run uses this to decide when the container is ready to serve traffic.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# ── Single uvicorn worker ─────────────────────────────────────────────────────
# Do NOT use multiple workers — the LLM singleton is not safe to fork.
# Scale via Cloud Run instances, not uvicorn workers.
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--timeout-keep-alive", "75", \
     "--log-level", "warning"]