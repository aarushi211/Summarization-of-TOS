# 📜 TOS-Summarizer: Distilled Legal AI

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![LlamaCPP](https://img.shields.io/badge/Llama_CPP-Quantized-orange)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-yellow)
![LangChain](https://img.shields.io/badge/Orchestration-LangChain-green)

## 🚀 Project Overview

Terms of Service (TOS) agreements are notoriously long and complex. This project creates a privacy-centric, efficient AI assistant capable of generating executive summaries and answering specific legal questions about TOS documents.

 [Live Demo on Google Cloud Run](https://tos-demo-110277869308.us-central1.run.app/) > (Note: The app runs on a scale-to-zero instance. Please allow ~30 seconds for the first cold start.)

## 🏗️ Architecture

This project implements a Hybrid RAG (Retrieval-Augmented Generation) architecture to solve the "Lost in the Middle" problem common in long legal documents.

```
graph TD
    User[User Uploads PDF] --> Ingest[PDF Ingestion & Cleaning]
    Ingest --> Router{Query Router}
    
    subgraph "Path A: Global Understanding"
    Router -->|Summarize Document| Context[Full Context Truncation]
    Context --> Model[Qwen 1.5B (Fine-Tuned)]
    Model --> Summary[Executive Summary]
    end
    
    subgraph "Path B: Specific Retrieval (RAG)"
    Router -->|Ask Question| VectorDB[FAISS Vector Store]
    VectorDB --> Retrieve[Retrieve Top-K Clauses]
    Retrieve --> Model
    Model --> Answer[Precise Legal Answer]
    end
```

## 🧠 Engineering Methodology

### Phase 1: Synthetic Data Pipeline (The "Teacher-Student" Loop)

High-quality, abstractive summaries for legal documents are scarce. To solve this, I engineered a synthetic data pipeline to generate a custom dataset.

**Source Data:** Used the raw text files from TOSDR (Terms of Service; Didn't Read) extracted by [Sonu Gupta](https://github.com/sonu-gupta/tosdr-terms-of-service-corpus).

**Teacher Model:** Llama 3.1 8B Instruct.

**Inference Engine:** Utilized vLLM (PagedAttention) on a T4 GPU to maximize throughput, generating 9,000 summaries.

**Outcome:** Created a high-quality, task-specific dataset mapping raw legal text to concise executive summaries.

**Alternative Teacher Models:** Gemini 2.5 Flash, llama3.1:8b

### Phase 2: Knowledge Distillation

Deploying a 7B+ model is expensive. I distilled the knowledge from the Teacher (Llama 3.1) into a Student model (Qwen 2.5 1.5B) to enable edge-friendly deployment.

**Technique:** Supervised Fine-Tuning (SFT) with QLoRA (4-bit quantization).

**Optimization:** Used gradient accumulation and checkpointing to fit training within Google Colab's free tier limits.

**Result:** A lightweight 3GB model that mimics the reasoning of Llama 3.1 but runs 4x faster.

### Phase 3: Quantization & Cloud Deployment

To make the system production-ready and cost-efficient:

**Quantization:** Converted the fine-tuned model to GGUF (4-bit) format using llama.cpp. This reduced the model size to ~1 GB, allowing it to run purely on CPU.

**Containerization:** Built a multi-stage Docker container optimized for Google Cloud Run.

**Infrastructure:** Configured custom memory limits (4GiB) and concurrency settings to serve the model serverlessly.

## 🛠️ Tech Stack

- LLM: Qwen 2.5 1.5B (Fine-Tuned), Llama 3.1 8B (Teacher)
- Inference: llama-cpp-python, vLLM
- Training: Hugging Face trl, peft (LoRA), bitsandbytes
- Backend/UI: Streamlit, Python 3.10
- DevOps: Docker, Google Cloud Run, Git LFS

## 📊 Performance Metrics

- Model Size Reduction: 15GB (FP16) $\rightarrow$ 1.2GB (GGUF Q4_K_M)
- Inference Speed: ~15 tokens/sec on standard CPU (vs. requiring GPU for original model).
- Qualitative Metrics: The model successfully identifies critical clauses (Arbitration, Data Selling) that generic base models often gloss over.
- Quantitative benchmarks (ROUGE/BERTScore) are currently being computed on a held-out test set.


## 💻 Local Installation
Want to run this offline?

**Clone the Repository**
```
git clone [https://github.com/yourusername/Summarization-of-TOS.git](https://github.com/yourusername/Summarization-of-TOS.git)
cd Summarization-of-TOS
```

**Install Dependencies**
```
# We recommend using a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

```

#### Run the App
```
streamlit run app/app.py
```