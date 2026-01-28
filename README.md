# 📜 TOS-Summarizer: Distilled Legal AI

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![LlamaCPP](https://img.shields.io/badge/Llama_CPP-Quantized-orange)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-yellow)
![LangChain](https://img.shields.io/badge/Orchestration-LangChain-green)

## 🚀 Project Overview

TOS-Summarizer is a production-ready legal AI system that generates **faithful executive summaries** and **clause-grounded answers** from long Terms of Service documents optimized for **low hallucination, low cost, and CPU-only deployment**.

Unlike generic RAG demos, this project tackles real-world legal challenges such as **lost-in-the-middle failures, hallucinations in compliance-critical text, and the cost of deploying large language models at scale**.

🔍 Designed for legal reliability.
⚙️ Engineered for deployment constraints.
🧠 Trained via synthetic data and knowledge distillation.

## Why This Project Matters

Terms of Service documents are:
- Extremely long and structurally dense
- Written in legally precise but non-user-friendly language
- Poorly handled by standard summarization and naive RAG pipelines

In practice, many LLM systems:
- Miss critical clauses (arbitration, data sharing, liability)
- Hallucinate obligations not present in the document
- Require large, expensive models to perform reliably

This project addresses these issues by combining **hybrid RAG, teacher-student distillation, and aggressive quantization**, resulting in a system that is both **legally cautious** and **operationally efficient**.

## 🧠 Key Technical Contributions

- Hybrid RAG Architecture: Separates global document understanding (executive summary) from targeted clause retrieval (question answering), mitigating lost-in-the-middle effects.
- Synthetic Data Generation at Scale: Built a teacher-student pipeline using Llama 3.1 8B to generate ~9,000 high-quality legal summaries, enabling supervised fine-tuning in a low-resource domain.
- Knowledge Distillation & Quantization: Distilled legal reasoning into a 1.5B-parameter model using QLoRA, then converted to 4-bit GGUF format to enable fast, CPU-only inference.
- Production-First Deployment: Deployed on Google Cloud Run with scale-to-zero, custom memory limits, and optimized containers — no GPU required.

## 🔗 Live Demo
[Live Demo on Google Cloud Run](https://tos-demo-110277869308.us-central1.run.app/) > (Note: The app runs on a scale-to-zero instance. Please allow ~1 min for the first cold start.)

## 🛠️ Tech Stack

- LLM: Qwen 2.5 1.5B (Fine-Tuned), Llama 3.1 8B (Teacher)
- Inference: llama-cpp-python, vLLM
- Training: Hugging Face trl, peft (LoRA), bitsandbytes
- Backend/UI: Streamlit, Python 3.10
- DevOps: Docker, Google Cloud Run, Git LFS

## 🏗️ Architecture

System Architecture (Hybrid RAG)

Long legal documents suffer from “Lost-in-the-Middle” failures, where critical clauses are ignored when context windows are saturated. To address this, the system uses a **Hybrid Retrieval-Augmented Generation (RAG)** architecture that explicitly separates:

- Global document understanding (executive summary generation)
- Targeted clause retrieval (question answering)

A lightweight query router determines the appropriate path at runtime, allowing the model to operate on either truncated global context or retrieved, clause-level evidence.

```
graph TD
    %% User Interaction
    U[User Uploads TOS PDF / Query] --> I[PDF Ingestion & Cleaning]

    %% Routing Logic
    I --> R{Query Router}
    R -->|Executive Summary| GS[Global Context Builder]
    R -->|Targeted Question| QR[Query Encoder]

    %% Global Summarization Path
    subgraph A[Path A: Global Document Understanding]
        GS --> TC[Context Truncation<br/>(Long-doc aware)]
        TC --> LLM1[Fine-Tuned Qwen 2.5 1.5B<br/>(GGUF, CPU)]
        LLM1 --> S[Executive Summary]
    end

    %% Retrieval-Augmented Path
    subgraph B[Path B: Clause-Grounded QA (RAG)]
        QR --> VDB[FAISS Vector Store]
        VDB --> RET[Top-K Clause Retrieval]
        RET --> LLM2[Fine-Tuned Qwen / Mistral<br/>(Grounded Inference)]
        LLM2 --> A[Faithful Answer<br/>(Low Hallucination)]
    end

    %% Outputs
    S --> O[User Output]
    A --> O
```

## 🧠 Engineering Methodology

### Phase 1: Synthetic Data Pipeline (The "Teacher-Student" Loop)

High-quality, abstractive summaries for legal documents are scarce. To solve this, I engineered a synthetic data pipeline to generate a custom dataset.

- **Source Data:** Used the raw text files from TOSDR (Terms of Service; Didn't Read) extracted by [Sonu Gupta](https://github.com/sonu-gupta/tosdr-terms-of-service-corpus).
- **Teacher Model:** Llama 3.1 8B Instruct.
- **Inference Engine:** Utilized vLLM (PagedAttention) on a T4 GPU to maximize throughput, generating 9,000+ summaries.
- **Outcome:** Created a high-quality, task-specific dataset mapping raw legal text to concise executive summaries.
- **Alternative Teacher Models:** Gemini 2.5 Flash, llama3.1:8b

### Phase 2: Knowledge Distillation

Deploying a 7B+ model is expensive. I distilled the knowledge from the Teacher (Llama 3.1) into a Student model (Qwen 2.5 1.5B) to enable edge-friendly deployment.

- **Technique:** Supervised Fine-Tuning (SFT) with QLoRA (4-bit quantization).
- **Optimization:** Used gradient accumulation and checkpointing to fit training within Google Colab's free tier limits.
- **Result:** A lightweight 3GB model that mimics the reasoning of Llama 3.1 but runs 4x faster.

### Phase 3: Quantization & Cloud Deployment

To make the system production-ready and cost-efficient:

- **Quantization:** Converted the fine-tuned model to GGUF (4-bit) format using llama.cpp. This reduced the model size to ~1 GB, allowing it to run purely on CPU.
- **Containerization:** Built a multi-stage Docker container optimized for Google Cloud Run.
- **Infrastructure:** Configured custom memory limits (4GiB) and concurrency settings to serve the model serverlessly.

## 📊 Evaluation & Analysis
Models were evaluated on a held-out test split from the synthetic dataset generated during the teacher–student pipeline. Each test sample consists of raw TOS text paired with a high-quality abstractive summary generated by a teacher LLM.

While synthetic, this setup ensures:
- Consistent supervision aligned with the task objective (executive legal summaries)
- Scalable and reproducible evaluation
- Fair comparison across student and baseline models

Given the legal domain’s emphasis on faithfulness and clause coverage, evaluation focused on multiple complementary metrics rather than a single score.

### Metrics Used

To capture different aspects of summarization quality, evaluation was performed across three dimensions:
- Lexical overlap: ROUGE-1/2/L, BLEU, METEOR
- Semantic similarity: BERTScore-F1
- Factual consistency: Faithfulness score, hallucination rate, and unsupported claims (evaluated on sampled outputs)

This multi-metric approach avoids over-optimizing for surface-level similarity and better reflects real-world legal reliability.

### Quantitative Results (Aggregate)

| Model                     | ROUGE-1   | ROUGE-2   | ROUGE-L   | BERTScore-F1 | BLEU      | METEOR    |
| ------------------------- | --------- | --------- | --------- | ------------ | --------- | --------- |
| **Qwen 2.5 1.5B**         | 0.364     | 0.130     | 0.218     | **0.849**    | 0.079     | 0.311     |
| **Qwen 2.5 GGUF (4-bit)** | **0.487** | **0.228** | **0.310** | 0.701        | **0.151** | **0.324** |
| **Mistral (FP16)**        | 0.457     | 0.226     | 0.309     | **0.868**    | 0.169     | **0.378** |
| **Mistral GGUF (4-bit)**  | **0.484** | **0.246** | **0.324** | 0.703        | **0.156** | 0.321     |

### Faithfulness & Hallucination Analysis

Faithfulness was evaluated using LLM as a judge, using Gemini-3-flash-preview  generated summaries by verifying whether each claim was directly supported by the source TOS text or retrieved clauses.

| Model            | Avg. Faithfulness | Hallucination Rate | Avg. Unsupported Claims |
| ---------------- | ----------------- | ------------------ | ----------------------- |
| **Qwen GGUF**    | 0.945             | 0.182              | 0.364                   |
| **Mistral GGUF** | **0.964**         | **0.071**          | **0.286**               |

### Key Observations & Insights

**Effect of Quantization**

Quantizing models to 4-bit GGUF format consistently:
- Improved ROUGE and BLEU scores, indicating stronger alignment with legally salient clauses
- Reduced hallucination rates, particularly for Mistral
- Slightly reduced BERTScore, suggesting less paraphrasing and more conservative, extractive behavior
- For legal summarization, this trade-off is desirable, as factual precision outweighs stylistic richness.

**Qwen vs Mistral Trade-off**
- Qwen GGUF provides an excellent balance of model size (~1 GB), inference speed, and clause recall, making it suitable for cost-sensitive, serverless deployments.
- Mistral GGUF achieves higher faithfulness and lower hallucination rates, making it better suited for higher-risk legal interpretations.

### Limitations
- Evaluation relies on synthetic teacher-generated summaries, which may inherit teacher biases.
- Faithfulness analysis was conducted on a sampled subset rather than the full test set.
- Jurisdiction-specific legal interpretations are not explicitly modeled.

Future work includes incorporating human-reviewed summaries and expanding faithfulness evaluation at scale.



## 💻 Local Installation
Want to run this offline?

**Clone the Repository**
```
git clone [https://github.com/aarushi211/Summarization-of-TOS.git](https://github.com/aarushi211/Summarization-of-TOS.git)
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

## ⚙️ Deployment & MLOps
Deploying LLMs to serverless infrastructure presents unique challenges regarding memory, build times, and cold starts. 

👉 **[Read the full Deployment Engineering Guide](DEPLOYMENT.md)** to see how I solved Docker build timeouts and optimized inference for CPU-only environments.