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

### Performance

#### Local benchmark (CPU, `RUNTIME_MODE=desktop`)

Measured with `python src/Evaluation/benchmark_latency.py` on 5 PDFs and 15
questions from `data/Test_data.csv` (FAISS + GGUF on CPU). Aggregates in
`latency_summary.csv` (2026-06-04 run).

| Stage | Latency |
|-------|---------|
| Model load (one-time) | 18.0s (GGUF 2.5s, embeddings 0.2s, cross-encoder 0.1s) |
| Document ingestion (avg) | 1.9s (PDF parse ~0.2s, FAISS index ~1.7s) |
| Global summary (avg) | 1m 07s (~9.6 tok/s) |
| Q&A per question (avg / median / P95) | 17.2s / 17.5s / 25.6s |
| Q&A breakdown (avg) | retrieval 1.3s, LLM 15.8s (~5.4 tok/s) |
| Full benchmark (5 docs + 15 Q&A) | 10m 21s |

#### Cloud Run (portfolio deployment)

Running with `min-instances=0` to minimize cost. Expect a cold start on first
request after inactivity; per-request latency is higher than local desktop due
to serverless CPU, GCS model mount, and cold caches.

| Operation | Latency |
|-----------|---------|
| Cold start | ~4m 45s |
| Document indexing | ~1m 06s |
| Q&A (per question) | ~1m 15s |
| Summary (per topic) | ~1m 30s |

Cold starts are intentional. Setting `min-instances=1` and baking models into
the image would reduce cold start to ~30s but increases monthly cost.
Tradeoffs documented in [DEPLOYMENT.md](./DEPLOYMENT.md).

---

## Architecture

### Core Inference Pipeline (`src/RAG/engine.py`)

- **PDF ingestion:** PyMuPDF primary loader with PyPDF fallback; text is
  cleaned with line breaks preserved and assembled with `<!-- page:N -->`
  markers before chunking
- **Structure-aware chunking:** in-memory markdown (not written to disk) →
  legal header splitting → recursive chunks (1200 / 300 overlap) with
  citation metadata (`page`, `section`, `citation`)
- **Hybrid retrieval:** dense vector search (Pinecone) fused with BM25 over
  the full chunk namespace via Reciprocal Rank Fusion (RRF), plus legal
  query expansion (`src/RAG/query_expansion.py`)
- **Cross-encoder reranking:** `ms-marco-MiniLM-L-6-v2` reranks top-50
  candidates to top-5 for Q&A (top-7 for longer documents)
- **Grounded Q&A:** extract-then-answer over retrieved chunks; abstention and
  notification guards; refusal path for legally dangerous prompts
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
src/Evaluation/      Quality metrics, red team, latency benchmark, RAGAS pipeline
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

## RAG Evaluation

Offline eval scripts in `src/Evaluation/` measure retrieval and generation
quality on `data/Test_data.csv` (15 questions across Netflix, Spotify,
YouTube, and OpenAI PDFs). Run from the repo root with models available
locally (see `DEPLOYMENT.md`). **Re-ingest documents** after chunking or
embedding changes so scores reflect the current index.

Set `GROQ_API_KEY` in `.env` for LLM-as-judge metrics (recommended). Without
it, faithfulness falls back to a local NLI model with more conservative scores.

| Metric | Score | Source CSV |
|--------|-------|------------|
| **Context recall** | **0.93** (14/15; 2 partial) | `context_recall_results.csv` |
| **Faithfulness** | **0.88** (QA mode, 15 questions) | `faithfulness_summary.csv` |
| **Answer relevance** | **0.60** (4/15 abstentions) | `answer_relevance_results.csv` |
| **Red team safety** | **100%** (9/9 safe) | `red_team_results.csv` |

Scores above are the latest run aggregates from those files at the repo root.
Judges use Groq (`llama-3.3-70b-versatile`) when `GROQ_API_KEY` is set.

Interpretation: retrieval is strong (only Spotify upload rules and OpenAI
automated-decision-making scored partial). The main gap is **Q&A abstention**
(four “I do not have enough information” answers despite high context recall on
some of those rows). Faithfulness stays high on abstentions because they add no
contradicted claims.

```bash
python src/Evaluation/context_recall.py
python src/Evaluation/faithfulness.py          # --mode qa | summary | both
python src/Evaluation/answer_relevance.py
python src/Evaluation/red_team.py
python src/Evaluation/benchmark_latency.py     # latency_summary.csv
```

Each script writes a CSV in the project root and prints a console summary.

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
| RAG quality metrics | `src/Evaluation/`, `data/Test_data.csv` |
| CI/CD rollout flow | `cloudbuild.yaml` |
| Deployment decisions | `DEPLOYMENT.md` |