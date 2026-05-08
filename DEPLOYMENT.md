# Deployment Guide

Backend on **Google Cloud Run**, frontend on **Vercel**,
model artifacts on **Google Cloud Storage**.

---

## 1. Architecture Overview

### Backend (Cloud Run)

- Containerized FastAPI app (`api/main.py`)
- Multi-stage Docker build: ~500MB runtime image (CPU-only PyTorch,
  no model files baked in)
- Model path at runtime: `/app/models/legal_qwen.Q4_K_M.gguf`
- Structured JSON logs with `X-Request-ID` correlation
- LangSmith tracing enabled in production (`LANGCHAIN_TRACING_V2=true`)

### Frontend (Vercel)

- Next.js app (`frontend/`)
- Calls Cloud Run backend over HTTPS
- CORS origin controlled by `ALLOWED_ORIGINS` env var on backend

### Model Delivery

Models are stored in a GCS bucket and mounted into Cloud Run at
`/app/models` via GCSFuse. This keeps the Docker image at ~500MB
instead of >3GB and allows model updates without image rebuilds.

Two deployment options are documented in the Dockerfile:

| Option | Image size | Cold start | Use case |
|--------|-----------|------------|----------|
| GCS mount (current) | ~500MB | ~4m 45s | Portfolio / low cost |
| Baked into image | >3GB | ~30s | High traffic / production |

---

## 2. Runtime Configuration

As configured in `cloudbuild.yaml`:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `cpu` | 4 | LLM inference is CPU-bound; more cores reduce token latency |
| `memory` | 8Gi | Model + embeddings + cross-encoder fit comfortably |
| `concurrency` | 1 | Single LLM process; concurrent requests would thrash |
| `timeout` | 300s | Long inference chains can take 60-90s per request |
| `min-instances` | 0 | Cost optimization for portfolio deployment |
| `max-instances` | 1 | Prevents runaway cost; single model process anyway |
| `--cpu-boost` | enabled | Allocates extra CPU during cold start initialization |
| `--execution-environment` | gen2 | Required for GCS volume mounts |

---

## 3. CI/CD Pipeline (`cloudbuild.yaml`)
1. Pull cache     → docker pull :latest (|| true — safe on first run)
2. Build          → docker build with --cache-from for layer reuse
3. Push           → push :$BUILD_ID and :latest tags
4. Deploy         → gcloud run deploy --no-traffic --tag candidate
5. Smoke test     → scripts/smoke-test.sh hits tagged revision URL
6. Migrate        → update-traffic --to-tags candidate=100

Key properties:
- `--no-traffic` on deploy means the new revision serves zero requests
  until smoke test passes
- Tagged revisions get a stable URL (`candidate---...run.app`) for
  testing without affecting live traffic
- If smoke test fails, Cloud Build exits non-zero and traffic migration
  never runs — old revision keeps serving

**Note:** `--no-traffic` fails if the Cloud Run service does not yet
exist. On first deploy to a new region, deploy manually once without
`--no-traffic`, then use Cloud Build for all subsequent deploys.

---

## 4. Secrets Management

All secrets stored in GCP Secret Manager, injected at runtime via
`--set-secrets`. Secret names match env var names 1:1.

Required secrets:
```
PINECONE_API_KEY
PINECONE_INDEX_NAME
SUPABASE_URL
SUPABASE_ANON_KEY
SUPABASE_SERVICE_KEY
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_S3_BUCKET_NAME
LANGCHAIN_API_KEY
ADMIN_SECRET
```

**Known gotcha:** Secrets created by pasting values in a terminal or
from files with trailing newlines will include `\r\n` in the stored
value. This causes HTTP header validation errors at runtime
(`Invalid header value`). Always verify with:

```bash
gcloud secrets versions access latest --secret=SECRET_NAME | xxd | tail -1
```

Clean value should show no `0d 0a` bytes at the end.

---

## 5. Common Failure Modes

### Container fails to start on port 8080

Cause: App crashes during lifespan startup before binding the port.

Check logs for the actual error before the traceback:
```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.revision_name="REVISION"' \
  --project=PROJECT_ID --limit=100 --order=asc \
  --format="table(timestamp,severity,textPayload)"
```

Common causes:
- Secret has trailing whitespace (`\r\n`) — see Section 4
- Secret value is wrong (wrong index name, invalid characters)
- Model file not found at `/app/models/...` — check GCS bucket contents
  and mount path

### `--no-traffic` fails on deploy

Cause: Service does not exist yet in that region.

Fix: Deploy manually once without `--no-traffic` to bootstrap the
service, then use Cloud Build for all future deploys.

### GCS volume mount fails

Cause: Missing `--execution-environment gen2`.

Fix: Ensure `gen2` is set in deploy args. Also confirm the Cloud Run
service account has `roles/storage.objectViewer` on the bucket.

### Smoke test fails to reach tagged revision

Cause: URL constructed incorrectly — tagged revision URL format differs
from service URL.

Fix: Derive the tagged URL from `gcloud run services describe` output,
not by constructing it manually.

---

## 6. Observability

**Cloud Logging** (structured JSON):
```bash
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=tos-api" \
  --project=tos-summarization \
  --limit=50 \
  --format="table(timestamp,severity,jsonPayload.message,jsonPayload.request_id)"
```

**LangSmith:** LLM call traces available at https://smith.langchain.com
under project `tos-summarizer-prod`.

**Request tracing:** Every request gets an `X-Request-ID` header
(visible in response). Use this to correlate frontend errors with
backend logs.

---

## 7. Cost vs UX Tradeoffs

| Mode | Config | Cold start | Monthly cost (est.) |
|------|--------|------------|---------------------|
| Portfolio (current) | min-instances=0 | ~4m 45s | ~$0-5 |
| Warmed | min-instances=1 | ~30s | ~$30-50 |
| Baked image + warmed | min-instances=1 + image bake | ~15s | ~$30-50 |

Current config accepts cold starts to keep costs near zero for a
portfolio deployment with infrequent traffic.

---

## 8. Frontend (Vercel)

- Set `NEXT_PUBLIC_API_URL` to your Cloud Run service URL
- Add Vercel production and preview domains to backend `ALLOWED_ORIGINS`
- Vercel handles CDN and static asset caching automatically

---

## 9. Re-deploying to a New Region

Checklist:
1. Create Artifact Registry repo in new region
2. Create GCS bucket in new region and copy model files
3. Grant Cloud Run service account `objectViewer` on new bucket
4. Replicate all secrets in Secret Manager (verify no trailing newlines)
5. Bootstrap with manual `gcloud run deploy` (no `--no-traffic`)
6. Update `cloudbuild.yaml` substitution defaults
7. Subsequent deploys via `gcloud builds submit`