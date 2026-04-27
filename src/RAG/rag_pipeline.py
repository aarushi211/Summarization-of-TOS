from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llama_cpp import Llama
from sentence_transformers import CrossEncoder
import re
import time
from langchain_community.document_loaders import TextLoader

class TOSAssistant:
    def __init__(self, model_path, index_dir="faiss_index"):
        self.index_dir = Path(index_dir)
        self._metrics = []

        # --- Time GGUF Model Load ---
        print(f"Loading RAG Model: {Path(model_path).name}")
        t0 = time.perf_counter()
        self.llm = Llama(
            model_path=model_path,
            n_ctx=8192,
            n_gpu_layers=-1,
            verbose=False
        )
        t_model = time.perf_counter() - t0
        self._metrics.append({"stage": "init", "sub_stage": "gguf_model_load", "latency_s": t_model})

        SCRIPT_DIR = Path(__file__).resolve().parent
        PROJECT_ROOT = SCRIPT_DIR.parent.parent
        local_embed_path = PROJECT_ROOT / "models" / "embeddings"
        local_cross_path = PROJECT_ROOT / "models" / "cross-encoder" / "ms-marco-MiniLM-L-6-v2"

        # --- Time Embedding Model Load ---
        print("Loading Embeddings & Cross-Encoder...")
        t0 = time.perf_counter()
        self.embed_model = HuggingFaceEmbeddings(
            model_name=str(local_embed_path),
            model_kwargs={"trust_remote_code": True}
        )
        t_embed = time.perf_counter() - t0
        self._metrics.append({"stage": "init", "sub_stage": "embedding_load", "latency_s": t_embed})

        # --- Time Cross-Encoder Load ---
        t0 = time.perf_counter()
        self.cross_encoder = CrossEncoder(str(local_cross_path))
        t_cross = time.perf_counter() - t0
        self._metrics.append({"stage": "init", "sub_stage": "cross_encoder_load", "latency_s": t_cross})

        total_init = t_model + t_embed + t_cross
        self._metrics.append({"stage": "init", "sub_stage": "total", "latency_s": total_init})

        self.vector_store = None
        self.full_text = ""
        self.doc_type = "Unkown Document"
        self.service_name = "Unknown Service"

    def clean_text(self, text):
        replacements = {
            "\u00ef\u00ac\u0081": "fi",
            "\u00ef\u00ac\u0082": "fl",
            "\u00ef\u00ac\u0080": "ff",
            "\u00ef\u00ac\u0083": "ffi",
            "\u00ef\u00ac\u0084": "ffl", 
            "": "",}
        
        for search, replace in replacements.items():
            text = text.replace(search, replace)

        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def ingest_document(self, pdf_path):
        print(f'Ingesting {pdf_path}...')
        metrics_entry = {"stage": "ingest_document", "document": str(pdf_path)}

        # --- PDF Parsing ---
        t0 = time.perf_counter()
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        metrics_entry["pdf_parse_s"] = time.perf_counter() - t0
        metrics_entry["page_count"] = len(documents)

        # --- Text Cleaning ---
        t0 = time.perf_counter()
        cleaned_pages = []
        for doc in documents:
            cleaned_content = self.clean_text(doc.page_content)
            cleaned_pages.append(cleaned_content)

        self.full_text = '\n'.join(cleaned_pages)

        for doc, clean_text in zip(documents, cleaned_pages):
            doc.page_content = clean_text
        metrics_entry["text_clean_s"] = time.perf_counter() - t0
        metrics_entry["total_chars"] = len(self.full_text)

        # --- Chunking ---
        t0 = time.perf_counter()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, 
            chunk_overlap=300
        )
        chunks = text_splitter.split_documents(documents)
        metrics_entry["chunking_s"] = time.perf_counter() - t0
        metrics_entry["chunk_count"] = len(chunks)
        
        # --- FAISS Indexing ---
        t0 = time.perf_counter()
        self.vector_store = FAISS.from_documents(chunks, self.embed_model)
        metrics_entry["faiss_index_s"] = time.perf_counter() - t0

        metrics_entry["total_ingest_s"] = (
            metrics_entry["pdf_parse_s"] + metrics_entry["text_clean_s"] +
            metrics_entry["chunking_s"] + metrics_entry["faiss_index_s"]
        )
        self._metrics.append(metrics_entry)
        print("Ingestion complete. Vector Store Ready.")

    def ingest_text_file(self, txt_path):
        print(f'Ingesting Text File {txt_path}...')
        metrics_entry = {"stage": "ingest_text", "document": str(txt_path)}

        # --- Text Loading ---
        t0 = time.perf_counter()
        loader = TextLoader(txt_path, encoding='utf-8')
        documents = loader.load()
        metrics_entry["text_load_s"] = time.perf_counter() - t0
        
        # --- Chunking ---
        t0 = time.perf_counter()
        # Reuse your existing splitter logic
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, 
            chunk_overlap=300
        )
        chunks = text_splitter.split_documents(documents)
        metrics_entry["chunking_s"] = time.perf_counter() - t0
        metrics_entry["chunk_count"] = len(chunks)

        # --- FAISS Indexing ---
        t0 = time.perf_counter()
        self.full_text = documents[0].page_content
        self.vector_store = FAISS.from_documents(chunks, self.embed_model)
        metrics_entry["faiss_index_s"] = time.perf_counter() - t0

        metrics_entry["total_chars"] = len(self.full_text)
        metrics_entry["total_ingest_s"] = (
            metrics_entry["text_load_s"] + metrics_entry["chunking_s"] +
            metrics_entry["faiss_index_s"]
        )
        self._metrics.append(metrics_entry)
        print("Text Ingestion complete.")

    def _get_relevant_chunks(self, query, top_k=5):
        retrieval_metrics = {}

        # --- MMR Search ---
        t0 = time.perf_counter()
        candidates = self.vector_store.max_marginal_relevance_search(
            query, k=50, fetch_k=100, lambda_mult=0.5
        )
        retrieval_metrics["mmr_search_s"] = time.perf_counter() - t0
        retrieval_metrics["mmr_candidates"] = len(candidates)

        # --- Cross-Encoder Reranking ---
        t0 = time.perf_counter()
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.cross_encoder.predict(pairs)
        scored_docs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        retrieval_metrics["rerank_s"] = time.perf_counter() - t0

        retrieval_metrics["total_retrieval_s"] = (
            retrieval_metrics["mmr_search_s"] + retrieval_metrics["rerank_s"]
        )

        result = [doc for doc, score in scored_docs[:top_k]]
        return result, retrieval_metrics

    def format_qwen_prompt(self, system_msg, user_msg):
        return (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def generate_global_summary(self):
        if not self.full_text:
            return "No document loaded."

        metrics_entry = {"stage": "global_summary"}

        total_len = len(self.full_text)
        samples = [
            self.full_text[:6000],          
            self.full_text[total_len//2-3000:total_len//2+3000],  
            self.full_text[-6000:]           
        ]
        
        truncated_text = "\n[...]\n".join(samples)
        metrics_entry["input_chars"] = len(truncated_text)

        system_prompt = "You are a legal expert. Summarize the following Terms of Service. Focus on user rights and data privacy."
        
        user_prompt = f"Service: {self.service_name}\nDoc Type: {self.doc_type}\n\nText:\n{truncated_text}"

        # --- Prompt Construction ---
        t0 = time.perf_counter()
        formatted_prompt = self.format_qwen_prompt(system_prompt, user_prompt)
        metrics_entry["prompt_construct_s"] = time.perf_counter() - t0

        # --- LLM Inference ---
        t0 = time.perf_counter()
        output = self.llm(
            formatted_prompt,
            max_tokens=500,
            temperature=0.1,
            repeat_penalty=1.2,
            stop=["<|im_end|>", "<|eot_id|>"],
            echo=False
        )
        metrics_entry["llm_inference_s"] = time.perf_counter() - t0

        # --- Extract token metrics from llama.cpp usage ---
        usage = output.get("usage", {})
        metrics_entry["prompt_tokens"] = usage.get("prompt_tokens", 0)
        metrics_entry["completion_tokens"] = usage.get("completion_tokens", 0)
        metrics_entry["total_tokens"] = usage.get("total_tokens", 0)
        if metrics_entry["llm_inference_s"] > 0 and metrics_entry["completion_tokens"] > 0:
            metrics_entry["tokens_per_sec"] = metrics_entry["completion_tokens"] / metrics_entry["llm_inference_s"]
        else:
            metrics_entry["tokens_per_sec"] = 0

        metrics_entry["total_summary_s"] = metrics_entry["prompt_construct_s"] + metrics_entry["llm_inference_s"]
        self._metrics.append(metrics_entry)

        return output['choices'][0]['text'].strip()

    def answer_question(self, query):
        if not self.vector_store:
            return "Please ingest a document first."

        metrics_entry = {"stage": "qa_inference", "query": query[:80]}

        # --- Retrieval (with internal timing) ---
        t0 = time.perf_counter()
        relevant_docs, retrieval_metrics = self._get_relevant_chunks(query, top_k=7)
        metrics_entry.update(retrieval_metrics)
        
        # --- Context Assembly ---
        context_str = "\n\n".join([
            f"[Source {i+1}]: {doc.page_content}" 
            for i, doc in enumerate(relevant_docs)
        ])
        metrics_entry["context_chars"] = len(context_str)

        system_msg = (
            "You are a strict legal assistant. Answer the user's question using ONLY the provided Context sources. "
            "Do not infer 'selling' of data unless the text explicitly states 'we sell data'. "
            "Distinguish between 'sharing' (for functionality) and 'selling' (for profit). "
            "If the answer is not present in the sources, say 'NOT_IN_DOCUMENT'."
        )
        
        user_msg = f"Context:\n{context_str}\n\nQuestion: {query}"

        # --- LLM Inference ---
        t0 = time.perf_counter()
        output = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.2,
            max_tokens=400,
            repeat_penalty=1.1,
            stop=["<|im_end|>"]
        )
        metrics_entry["llm_inference_s"] = time.perf_counter() - t0

        # --- Extract token metrics from llama.cpp usage ---
        usage = output.get("usage", {})
        metrics_entry["prompt_tokens"] = usage.get("prompt_tokens", 0)
        metrics_entry["completion_tokens"] = usage.get("completion_tokens", 0)
        metrics_entry["total_tokens"] = usage.get("total_tokens", 0)
        if metrics_entry["llm_inference_s"] > 0 and metrics_entry["completion_tokens"] > 0:
            metrics_entry["tokens_per_sec"] = metrics_entry["completion_tokens"] / metrics_entry["llm_inference_s"]
        else:
            metrics_entry["tokens_per_sec"] = 0

        metrics_entry["total_qa_s"] = metrics_entry["total_retrieval_s"] + metrics_entry["llm_inference_s"]
        self._metrics.append(metrics_entry)

        return {
            "answer": output['choices'][0]['message']['content'],
            "sources": [d.page_content[:200] + "..." for d in relevant_docs]
        }

    def get_metrics(self):
        """Return a copy of all accumulated timing metrics."""
        return list(self._metrics)

    def reset_metrics(self):
        """Clear all accumulated metrics (keeps init metrics)."""
        init_metrics = [m for m in self._metrics if m.get("stage") == "init"]
        self._metrics = init_metrics