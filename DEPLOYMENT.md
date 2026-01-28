# ☁️ Deployment & MLOps Strategy
This document details the engineering challenges faced and solutions implemented while deploying the TOS-Summarizer to a serverless environment (Google Cloud Run) and Hugging Face Spaces with strict resource constraints.

## 1. The Inference Engine Challenge (Docker & C++)
### The Problem
The project uses `llama.cpp` for efficient inference. However, installing the Python bindings (`llama-cpp-python`) requires compiling C++ code during the Docker build process.
- **Issue:** On Google Cloud Build (and Hugging Face Spaces free tier), compiling from source took >15 minutes, causing build timeouts.
- **Error:** Job failed with exit code: `1. Reason: Job timeout`.

### The Solution: Pre-built Wheels
Instead of compiling from source, I optimized the `Dockerfile` to fetch pre-compiled binary wheels compatible with the Linux container environment.

**Before (Timeouts):**
```
RUN apt-get install build-essential
RUN pip install llama-cpp-python  # Triggers compilation
```

**After (Builds in <2 mins):**
```
# Use pre-built binary for CPU
RUN pip install llama-cpp-python \
    --extra-index-url [https://abetlen.github.io/llama-cpp-python/whl/cpu](https://abetlen.github.io/llama-cpp-python/whl/cpu)
```

## 2. Resource Constraints (RAM & Quantization)
### The Problem
Standard serverless instances (Cloud Run Gen 1) default to 512MB - 1GB RAM.
- **Mistral 7B (FP16):** ~15 GB VRAM. Impossible to deploy cheaply.
- **Mistral 7B (Quantized Q8):** ~7.5 GB RAM. Still requires expensive custom instances or GPU tiers.

### The Solution: Aggressive Distillation & Quantization
I implemented a pipeline to distill knowledge into a smaller architecture (Qwen 1.5B) and quantized it to 4-bit GGUF.
| Model | Format | Size | Deployment Status |
| ------ | ------- | ------- | --------------|
| Mistral 7B | GGUF (Q8) | ~7.5 GB | Local Only (Too heavy for free tier hosting) |
| Qwen 1.5B | GGUF (Q4_K_M) | < 1 GB | Deployed (Fits easily in 2GB container) |

By pivoting to the 1.5B model, I reduced the artifact size by ~87%, allowing deployment on standard CPU instances without crashing.

## 3. Cold Starts vs. Cost Optimization
### The Trade-off
Serverless containers "scale to zero" when idle. The first user request triggers a "Cold Start," where the container boots up and loads the model into memory.
- **Optimized for Performance:** Setting `--min-instances 1` keeps the container warm (latency <5s) but consumes cloud credits 24/7.
- **Optimized for Cost:** Setting `--min-instances 0` kills the container when unused.

### Final Decision: Scale to Zero
For this portfolio project, I prioritized cost efficiency. I configured the service to scale to zero.
- **Trade-off:** The first request takes ~30-40 seconds to spin up the container and load the model.
- **Benefit:** Zero cost when the app is idle.

## 4. Deployment Architecture
I utilized a dual-deployment strategy to ensure accessibility:

1. **Hugging Face Spaces (Demo):**
    - Hosts an earlier checkpoint of the Qwen 1.5B model.
    - Serves as a permanent, always-available public demo.

2. **Google Cloud Run (Production):**
    - Hosts the latest fine-tuned Qwen 1.5B model.
    - Demonstrates a production-ready Dockerized Streamlit App.

## 🛠️ Command Cheat Sheet

This section serves as a runbook for building, testing, and deploying the application.
> Note: Deployment configurations (such as cloudbuild.yaml and production Dockerfiles) are maintained in the deploy branch. Please checkout that branch before executing deployment commands.

### 1. Local Docker Testing
Before deploying, validate the container locally to debug port mappings and model loading.
```
# Build the image locally
# -f app/Dockerfile: Use the specific Cloud Run config
# -t tos-local: Tag it with a name
docker build -f app/Dockerfile -t tos-local .

# Run the container
# Maps laptop port 9000 -> container port 8080
docker run -p 9000:8080 tos-local

# Access App: http://localhost:9000

# Debugging Commands
docker ps                       # See running containers
docker logs <CONTAINER_ID>      # Check for Python/Model errors
docker rm -f <CONTAINER_ID>     # Force stop a container
```


### 2. Google Cloud Setup (One-Time)
Initialize the project and enable required serverless components.
```
# 1. Login & Set Project
gcloud auth login
gcloud config set project tos-summarization

# 2. Enable APIs (Compute, Registry, Build)
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

# 3. Create Artifact Repository
gcloud artifacts repositories create tos-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker repository for TOS Summarizer"

# 4. Authenticate Docker
gcloud auth configure-docker us-central1-docker.pkg.dev
```


### 3. Production Deployment (Repeatable)
Deploy updates to the live service.

**Step A: Build & Push Image**

Using Cloud Build avoids local bandwidth bottlenecks when uploading large model files.
```
# Direct Build
gcloud builds submit --file app/Dockerfile --tag us-central1-docker.pkg.dev/[PROJECT_ID]/tos-repo/tos-streamlit:v1 .

# Alternative: Using cloudbuild.yaml
# gcloud builds submit --config cloudbuild.yaml .
```


**Step B: Deploy Service**

Deploy with custom resource limits for NLP workloads.
```
gcloud run deploy tos-demo \
    --image us-central1-docker.pkg.dev/[PROJECT_ID]/tos-repo/tos-streamlit:v1 \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --port 8080 \
    --memory 4Gi \
    --cpu 2 \
    --min-instances 0
```


**Step C: Cost Management**
If testing min-instances 1 for a demo, use this to reset it to 0 (Cold Start mode) to stop credit consumption.
```
gcloud run services update tos-demo --min-instances 0 --region us-central1
```