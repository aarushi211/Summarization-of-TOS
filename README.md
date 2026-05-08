# TOS Summarizer

An end-to-end AI engineering project for legal document analysis, built to production standards as an architectural showcase.

Designed to demonstrate: **Advanced RAG Pipelines**, **Secure API Design**, and **Cloud-Native Deployment** at scale.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Frontend-Next.js-black)](https://nextjs.org/)
[![Cloud Run](https://img.shields.io/badge/Deploy-Google%20Cloud%20Run-4285F4)](https://cloud.google.com/run)
[![Vercel](https://img.shields.io/badge/Frontend-Vercel-000000)](https://vercel.com/)

> **Disclaimer:** This project is for technical demonstration only. Not legal advice.

---

## What This Project Demonstrates

Built to production engineering standards, with a focus on the problems that matter in real legal-tech deployments

- Long-document summarization and grounded Q&A with hybrid retrieval and reranking
- Production-style API engineering: structured logging, request tracing, rate limiting, auth
- Security-focused ingestion: file validation and SSRF-safe URL resolution
- Serverless GCP deployment with tagged rollout, smoke testing, and zero-downtime traffic migration
- End-to-end delivery with a modern Next.js frontend

---

## Live Deployment

- **Frontend:** https://tos-summarization.vercel.app
- **Backend:** https://tos-api-110277869308.us-east1.run.app

### Performance (Portfolio Deployment)

Running on Cloud Run with `min-instances=0` to minimize cost. Expect a cold
start on first request after inactivity.

| Operation              | Latency   |
|------------------------|-----------|
| Cold start             | ~4m 45s   |
| Document indexing      | ~1m 06s   |
| Q&A (per question)     | ~1m 15s   |
| Summary (per topic)    | ~1m 30s   |

Cold starts are intentional. Setting `min-instances=1` and baking models into
the image would reduce cold start to ~30s but increases monthly cost.
Tradeoff documented in [DEPLOYMENT.md](./DEPLOYMENT.md).

---

## Architecture

### Core Inference Pipeline (`src/RAG/engine.py`)

- **Structure-aware chunking:** PDF → markdown → legal header splitting →
  recursive chunking with citation metadata (`page`, `section`, `citation`)
- **Hybrid retrieval:** dense vector search (Pinecone) fused with BM25 via
  Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking:** `ms-marco-MiniLM-L-6-v2` reranks top-50
  candidates to top-7
- **Streaming responses:** token-level SSE streaming to client
- **Evidence attribution:** every response returns source objects with
  `citation`, `section`, `page`, and `excerpt`
- **Model runtime:** quantized GGUF inference via `llama-cpp-python` on
  CPU-only Cloud Run

### Platform Design

- **Backend:** FastAPI (`api/`) with routers for auth, ingest, chat, history, admin
- **Frontend:** Next.js (`frontend/`)
- **Vector store:** Pinecone (server) / ChromaDB (desktop/local)
- **Database:** Supabase
- **File storage:** AWS S3
- **Model delivery:** GCS bucket mounted into Cloud Run at `/app/models`
- **Infra:** Cloud Build → Artifact Registry → Cloud Run

---

## Engineering Highlights

**Structured logging + request tracing**
Request IDs injected via middleware, propagated through logs and
`X-Request-ID` response headers. Integrated with LangSmith for LLM
call tracing.

**Multi-stage Docker build**
Builder stage compiles dependencies (including CPU-only PyTorch at
~500MB vs ~2GB GPU). Runtime stage copies only compiled packages,
keeping the final image ~500MB. Model files mounted at runtime via
GCS, not baked in.

**SSRF-safe URL validation (`api/utils/validation.py`)**
Hostname resolution with private IP range blocking before any
outbound fetch. Prevents server-side request forgery on document
URL ingestion.

**Deployment safety (`cloudbuild.yaml`)**
Deploy with `--no-traffic`, smoke-test the tagged `candidate`
revision, then migrate 100% traffic only on success. Zero-downtime
rollouts with automatic rollback on smoke test failure.

**Auth + rate limiting**
Bearer token auth with user-aware SlowAPI rate limiter. Per-user
key strategy prevents single-user exhaustion.

**Desktop/server abstraction**
Pinecone ↔ ChromaDB and Supabase ↔ SQLite swap via
`RUNTIME_MODE=desktop`. Lets the full pipeline run locally
without any cloud dependencies.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11, TypeScript |
| Backend | FastAPI, Pydantic, Uvicorn, SlowAPI |
| LLM runtime | llama-cpp-python (GGUF, CPU) |
| Retrieval | LangChain, Pinecone, BM25, CrossEncoder |
| Embeddings | HuggingFace (local, sentence-transformers) |
| Database | Supabase (PostgreSQL) |
| File storage | AWS S3 |
| Frontend | Next.js 15, React 19 |
| Containerization | Docker multi-stage |
| CI/CD | Google Cloud Build |
| Hosting | Cloud Run (backend), Vercel (frontend) |
| Observability | LangSmith, Cloud Logging |

---

## Repository Structure

```text
api/                 FastAPI app, routers, config, security, middleware
src/RAG/             Retrieval and generation pipeline
frontend/            Next.js frontend
tests/               API and RAG pipeline tests
cloudbuild.yaml      CI/CD: build → push → deploy → smoke test → migrate
DEPLOYMENT.md        Deployment decisions, failure modes, cost tradeoffs
```

---

## Local Development

### Backend

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r api/requirements.prod.txt
uvicorn api.main:app --reload --host 127.0.0.1 --port 8080
```

For desktop mode (no cloud dependencies):
```bash
RUNTIME_MODE=desktop uvicorn api.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Required Environment Variables (server mode)
```
PINECONE_API_KEY
PINECONE_INDEX_NAME
SUPABASE_URL
SUPABASE_SERVICE_KEY
SUPABASE_ANON_KEY
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_S3_BUCKET_NAME
ADMIN_SECRET
ALLOWED_ORIGINS
```
---

## Tests

```bash
pytest tests/ -q
```

Covers API health, auth, upload validation, and RAG pipeline behavior.

---

## Deployment

Full deployment via Cloud Build:

```bash
gcloud builds submit --config cloudbuild.yaml .
```

See [DEPLOYMENT.md](./DEPLOYMENT.md) for the full pipeline breakdown,
failure modes, and cost/UX tradeoffs.

---

## Recruiter / Engineer Entry Points

| What you want to see | Where to look |
|---|---|
| App lifecycle + observability | `api/main.py` |
| SSRF + upload security | `api/utils/validation.py` |
| Auth + rate limiting | `api/core/security.py` |
| RAG pipeline (RRF + reranking) | `src/RAG/engine.py` |
| CI/CD rollout flow | `cloudbuild.yaml` |
| Deployment decisions | `DEPLOYMENT.md` |