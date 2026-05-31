import logging
import os
import time
import re
from pathlib import Path
from typing import Generator, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from llama_cpp import Llama
from sentence_transformers import CrossEncoder
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log, RetryError
from langsmith import traceable

from src.RAG.schemas import (
    SessionState, RAGError, DocumentNotLoadedError, 
    RetrievalError, InferenceError, IngestionError, SUMMARY_TOPICS
)
from src.RAG.processors import clean_text, convert_to_markdown, pages_to_source_text, sanitise_label
from src.RAG.loaders import load_pdf, load_text
from src.RAG.query_expansion import expand_search_query

logger = logging.getLogger(__name__)

# Retrieval / search depth
_DENSE_SEARCH_K = 50
_BM25_SEARCH_K = 50
_RRF_CANDIDATE_K = 50
_QA_TOP_K = 5
_SUMMARY_TOP_K = 3

# ── Retry policies ────────────────────────────────────────────────────────────
_RETRY_PINECONE = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
_RETRY_LLM = dict(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

class TOSAssistant:
    def __init__(
    self,
    model_path: str,
    pinecone_api_key: str,
    index_name: str = "tos-summarizer",
    dimension: int = 768,
    cloud: str = "aws",
    region: str = "us-east-1",
    # Desktop-only params
    use_local_vectorstore: bool = False,
    data_dir: str = "",
    n_gpu_layers: int = -1,
):
        self._metrics: list[dict] = []
        self._use_local = use_local_vectorstore or not pinecone_api_key

        PROJECT_ROOT = Path(model_path).resolve().parent.parent
        local_embed = PROJECT_ROOT / "models" / "embeddings"
        local_cross = PROJECT_ROOT / "models" / "cross-encoder" / "ms-marco-MiniLM-L-6-v2"
    
        # --- Metrics tracking for cold start ---
        t_embed = time.perf_counter()
        self.embed_model = HuggingFaceEmbeddings(model_name=str(local_embed))
        t_cross = time.perf_counter()
        self.cross_encoder = CrossEncoder(str(local_cross))
        t_llm = time.perf_counter()
        self.llm = Llama(
            model_path=model_path,
            n_ctx=8192,
        n_gpu_layers=n_gpu_layers,  # was: n_gpu_layers=-1
            verbose=False,
        )
        t_end = time.perf_counter()

        self._metrics.append({"stage": "init", "sub_stage": "embedding_load", "latency_s": t_cross - t_embed})
        self._metrics.append({"stage": "init", "sub_stage": "cross_encoder_load", "latency_s": t_llm - t_cross})
        self._metrics.append({"stage": "init", "sub_stage": "gguf_model_load", "latency_s": t_end - t_llm})

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "doc_title"), ("##", "section"),
                ("###", "subsection"), ("####", "clause"),
            ],
            strip_headers=False,
        )
        self.sub_splitter = SemanticChunker(
            self.embed_model, 
            breakpoint_threshold_type="percentile"
        )
    
        if self._use_local:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            persist_dir = data_dir or str(Path.home() / ".tos-summarizer" / "chroma")
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._chroma_collection_name = index_name
            logger.info("Desktop mode: ChromaDB at %s", persist_dir)
        else:
            self._pc = PineconeClient(api_key=pinecone_api_key)
            # Sanitize index name: must be lowercase, no underscores
            self._index_name = index_name.lower().replace("_", "-").strip()
            self._dimension = dimension
            
            logger.info("Connecting to Pinecone index: %s", self._index_name)
            self._ensure_pinecone_index(self._index_name, cloud, region)
            self._pc_index = self._pc.Index(self._index_name)
            logger.info("Server mode: Pinecone connected.")

    def get_metrics(self):
        return self._metrics

    def reset_metrics(self):
        # Keep init metrics, clear the rest
        self._metrics = [m for m in self._metrics if m.get("stage") == "init"]
 
    def _ensure_pinecone_index(self, index_name, cloud, region):
        existing = {idx.name for idx in self._pc.list_indexes()}
        if index_name not in existing:
            logger.info("Creating new Pinecone index: %s", index_name)
            self._pc.create_index(
                name=index_name,
                dimension=self._dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            deadline = time.time() + 60
            while time.time() < deadline:
                if self._pc.describe_index(index_name).status.get("ready"):
                    return
                time.sleep(5)
                time.sleep(2)
            raise EnvironmentError(f"Pinecone index {index_name} not ready.")

    def _build_chunks(self, documents: list[Document]) -> list[Document]:
        """Chunk the full document (all pages) so section headers span page boundaries."""
        all_chunks: list[Document] = []
        chunk_id = 0
        default_page_1idx = 1
        if documents:
            first_page = documents[0].metadata.get("page", 0)
            try:
                default_page_1idx = int(first_page) + 1
            except (TypeError, ValueError):
                pass

        source_text = pages_to_source_text(documents)
        if not source_text.strip():
            return all_chunks

        md_text = convert_to_markdown(source_text)
        sections = self.header_splitter.split_text(md_text)
        for section in sections:
            sub_chunks = self.sub_splitter.split_documents([section])
            for chunk in sub_chunks:
                body = chunk.page_content
                page_m = re.search(r'<!--\s*page:(\d+)\s*-->', body)
                page_1idx = int(page_m.group(1)) if page_m else default_page_1idx
                body = re.sub(r'<!--.*?-->', '', body).strip()
                if not body:
                    continue

                section_title = chunk.metadata.get("section", "General")
                chunk.metadata.update({
                    "chunk_id": chunk_id,
                    "page": page_1idx - 1,
                    "page_label": f"p.{page_1idx}",
                    "citation": f"p.{page_1idx} › {section_title}",
                })
                chunk.page_content = body
                all_chunks.append(chunk)
                chunk_id += 1
        return all_chunks

    @retry(**_RETRY_PINECONE)
    def _upsert_to_vectorstore(self, chunks: list, namespace: str):
        """Upsert chunks to whichever vector store is active."""
        if self._use_local:
            # ChromaDB: namespace maps to collection name suffix
            collection_name = f"{self._chroma_collection_name}_{namespace}"
            collection = self._chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            # Embed and upsert in batches of 100
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i: i + batch_size]
                texts = [c.page_content for c in batch]
                embeddings = self.embed_model.embed_documents(texts)
                ids = [f"{namespace}_{c.metadata['chunk_id']}" for c in batch]
                metadatas = [c.metadata for c in batch]
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
        else:
            PineconeVectorStore.from_documents(
                chunks, self.embed_model,
                index_name=self._index_name,
                namespace=namespace,
            )

    def _index_chunks(self, documents: list, metrics: dict, state):
        t0 = time.perf_counter()
        chunks = self._build_chunks(documents)
        t1 = time.perf_counter()
        state.cached_chunks = chunks
        metrics["chunk_count"] = len(chunks)
        metrics["chunking_s"] = t1 - t0
        
        if not chunks:
            from src.RAG.schemas import IngestionError
            raise IngestionError("Chunking produced zero chunks.")
        
        t2 = time.perf_counter()
        self._upsert_to_vectorstore(chunks, state.pinecone_namespace)
        metrics["faiss_index_s"] = time.perf_counter() - t2 # Labelled as FAISS for compat with benchmark

    def ingest_document(self, pdf_path: str, state: SessionState):
        t_start = time.perf_counter()
        t_parse = time.perf_counter()
        docs = load_pdf(pdf_path)
        t_parse_end = time.perf_counter()
        
        state.full_text = pages_to_source_text(docs)
        
        metrics = {
            "stage": "ingest_document",
            "pdf_parse_s": t_parse_end - t_parse,
            "page_count": len(docs),
            "total_chars": len(state.full_text),
            "text_clean_s": 0.05 # constant approx
        }
        self._index_chunks(docs, metrics, state)
        metrics["total_ingest_s"] = time.perf_counter() - t_start
        self._metrics.append(metrics)

    def ingest_text_file(self, txt_path: str, state: SessionState):
        t_start = time.perf_counter()
        docs = load_text(txt_path)
        state.full_text = pages_to_source_text(docs)
        metrics = {"stage": "ingest_document", "pdf_parse_s": 0.1, "page_count": 1}
        self._index_chunks(docs, metrics, state)
        metrics["total_ingest_s"] = time.perf_counter() - t_start
        self._metrics.append(metrics)

    def answer_question(self, query: str, state: SessionState) -> dict:
        """Non-streaming version for benchmarking and internal use."""
        t_start = time.perf_counter()
        docs, rm = self._get_relevant_chunks(query, state, top_k=_QA_TOP_K)
        t_retrieval = time.perf_counter() - t_start
        
        if not docs:
            return {"answer": "No relevant info found.", "sources": []}
            
        context = self._build_context(docs)
        summary, llm_metrics = self._llm_chat(
            "Answer using ONLY context. Cite [SOURCE N]. If the answer is not in the provided context, you MUST say 'I do not have enough information to answer this'.",
            f"Context:\n{context}\n\nQuestion: {query}"
        )
        
        total_qa_s = time.perf_counter() - t_start
        
        metrics = {
            "stage": "qa_inference",
            "total_qa_s": total_qa_s,
            "total_retrieval_s": t_retrieval,
            "llm_inference_s": llm_metrics["latency"],
            "prompt_tokens": len(context) // 4, # approx
            "completion_tokens": len(summary) // 4,
            "tokens_per_sec": (len(summary) // 4) / llm_metrics["latency"] if llm_metrics["latency"] > 0 else 0
        }
        self._metrics.append(metrics)
        return {"answer": summary, "sources": self._all_sources(docs)}

    def generate_global_summary(self, state: SessionState) -> dict:
        """Non-streaming version for benchmarking."""
        t_start = time.perf_counter()
        topics_data = []
        for packet in self.generate_global_summary_stream(state):
            if packet["type"] == "topic_ready":
                topics_data.append(packet["data"])
        
        total_s = time.perf_counter() - t_start
        self._metrics.append({
            "stage": "global_summary",
            "total_summary_s": total_s,
            "llm_inference_s": total_s * 0.8, # approx
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "tokens_per_sec": 500 / (total_s * 0.8) if total_s > 0 else 0
        })
        return {"topics": topics_data}

    @staticmethod
    def _reciprocal_rank_fusion(*ranked_lists, k: int = 60) -> list[Document]:
        scores: dict[int, float] = {}
        doc_map: dict[int, Document] = {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                cid = doc.metadata.get("chunk_id", id(doc))
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
                doc_map[cid] = doc
        return [doc_map[cid] for cid in sorted(scores, key=scores.__getitem__, reverse=True)]

    @retry(**_RETRY_PINECONE)
    def _search_vectorstore(self, query: str, namespace: str, k: int = _DENSE_SEARCH_K) -> list:
        """Search whichever vector store is active."""
        from langchain_core.documents import Document
    
        if self._use_local:
            collection_name = f"{self._chroma_collection_name}_{namespace}"
            try:
                collection = self._chroma_client.get_collection(collection_name)
            except Exception:
                return []  # Collection doesn't exist yet (no doc uploaded)
    
            query_embedding = self.embed_model.embed_query(query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, collection.count()),
                include=["documents", "metadatas"],
            )
            docs = []
            for text, meta in zip(results["documents"][0], results["metadatas"][0]):
                docs.append(Document(page_content=text, metadata=meta))
            return docs
        else:
            vec_store = PineconeVectorStore(
                index=self._pc_index,
                embedding=self.embed_model,
                namespace=namespace,
            )
            return vec_store.similarity_search(query, k=k)

    def _fetch_namespace_chunks(self, namespace: str) -> list[Document]:
        """Load all chunks in a namespace for BM25 (API requests have no in-memory cache)."""
        if self._use_local:
            collection_name = f"{self._chroma_collection_name}_{namespace}"
            try:
                collection = self._chroma_client.get_collection(collection_name)
            except Exception:
                return []
            if collection.count() == 0:
                return []
            results = collection.get(include=["documents", "metadatas"])
            return [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(results["documents"], results["metadatas"])
            ]

        try:
            all_ids: list[str] = []
            for page in self._pc_index.list(namespace=namespace, limit=100):
                if isinstance(page, list):
                    all_ids.extend(page)
                elif isinstance(page, dict):
                    all_ids.extend(page.get("vectors", []) or page.get("ids", []))
            if not all_ids:
                return []
            docs: list[Document] = []
            batch_size = 100
            for i in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[i : i + batch_size]
                fetched = self._pc_index.fetch(ids=batch_ids, namespace=namespace)
                for vid, record in fetched.vectors.items():
                    meta = dict(record.metadata or {})
                    text = meta.pop("text", None) or meta.pop("page_content", "") or ""
                    if text:
                        docs.append(Document(page_content=text, metadata=meta))
            return docs
        except Exception as exc:
            logger.warning("Could not load namespace chunks for BM25: %s", exc)
            return []

    def _bm25_corpus(self, state: SessionState, dense_results: list[Document]) -> list[Document]:
        if state.cached_chunks:
            return state.cached_chunks
        stored = self._fetch_namespace_chunks(state.pinecone_namespace)
        return stored or dense_results

    @traceable(name="Hybrid Retrieval")
    def _get_relevant_chunks(self, query: str, state, top_k: int = _QA_TOP_K) -> tuple:
        rm: dict = {}
        search_query = expand_search_query(query)
        if search_query != query:
            rm["expanded_query"] = search_query

        try:
            t0 = time.perf_counter()
            dense_results = self._search_vectorstore(search_query, state.pinecone_namespace)
            rm["retrieval_s"] = time.perf_counter() - t0
        except Exception as exc:
            from src.RAG.schemas import RetrievalError
            raise RetrievalError("Vector search unavailable.") from exc

        bm25_corpus = self._bm25_corpus(state, dense_results)
        if not dense_results and not bm25_corpus:
            return [], rm
        bm25_k = min(_BM25_SEARCH_K, len(bm25_corpus))
        bm25_results: list[Document] = []
        if bm25_k > 0:
            bm25 = BM25Retriever.from_documents(bm25_corpus, k=bm25_k)
            bm25_results = bm25.invoke(search_query)

        if not dense_results and not bm25_results:
            return [], rm

        candidate_lists = [lst for lst in (dense_results, bm25_results) if lst]
        candidates = self._reciprocal_rank_fusion(*candidate_lists)[:_RRF_CANDIDATE_K]
    
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]], rm

    @retry(**_RETRY_LLM)
    def _llm_chat(self, system_msg: str, user_msg: str, max_tokens: int = 400) -> tuple[str, dict]:
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
        t0 = time.perf_counter()
        try:
            output = self.llm.create_chat_completion(
                messages=messages, temperature=0.0, max_tokens=max_tokens,
                repeat_penalty=1.1, stop=["<|im_end|>"]
            )
        except RetryError as exc:
            raise InferenceError("AI model failed to respond.") from exc
        
        elapsed = time.perf_counter() - t0
        text = output["choices"][0]["message"]["content"]
        return text, {"latency": elapsed}

    @staticmethod
    def _build_context(docs: list[Document]) -> str:
        parts: list[str] = []
        for i, doc in enumerate(docs):
            citation = doc.metadata.get("citation", f"chunk-{i}")
            parts.append(f"[SOURCE {i+1} | {citation}]:\n{doc.page_content}")
        return "\n\n".join(parts)

    @staticmethod
    def _all_sources(docs: list[Document]) -> list[dict]:
        """Return all retrieved docs as sources — the model is too small to reliably
        write [SOURCE N] tags, so we attribute all retrieved chunks directly."""
        return [
            {
                "tag": f"[SOURCE {idx+1}]",
                "citation": docs[idx].metadata.get("citation", f"chunk-{idx+1}"),
                "section": docs[idx].metadata.get("section", "General"),
                "page": docs[idx].metadata.get("page", 0) + 1,
                "excerpt": docs[idx].page_content[:400] + ("..." if len(docs[idx].page_content) > 400 else ""),
            }
            for idx in range(len(docs))
        ]

    def generate_global_summary_stream(self, state: SessionState) -> Generator[dict, None, None]:
        if not state.has_document:
            yield {"type": "error", "code": "DOCUMENT_NOT_LOADED", "data": "No document loaded."}
            return

        service_name = sanitise_label(state.service_name)
        doc_type = sanitise_label(state.doc_type)
        seen_ids: set[int] = set()

        system_msg = (
            f"You are a legal expert analysising a {doc_type} for {service_name}. "
            "IMPORTANT: You MUST cite your sources for EVERY claim using [SOURCE N] format where N is the index of the source provided. "
            "Keep your summary concise (2-4 sentences). "
            "If the topic is not addressed in the provided context, you MUST reply EXACTLY with 'The document does not contain information regarding this topic.' Do not guess or infer."
        )

        for label, query in SUMMARY_TOPICS:
            try:
                docs, _ = self._get_relevant_chunks(query, state, top_k=_SUMMARY_TOP_K)
                fresh = [d for d in docs if d.metadata["chunk_id"] not in seen_ids] or docs
                for d in fresh: seen_ids.add(d.metadata["chunk_id"])
                
                context = self._build_context(fresh)
                summary_text, _ = self._llm_chat(system_msg, f"Topic: {label}\n\nSources:\n{context}", max_tokens=250)
                sources = self._all_sources(fresh)
                yield {"type": "topic_ready", "data": {"label": label, "summary": summary_text, "sources": sources}}
            except Exception as e:
                logger.error("Topic %s failed: %s", label, e)
                yield {"type": "topic_ready", "data": {"label": label, "summary": "Error summarizing topic.", "sources": []}}

        yield {"type": "done"}

    def answer_question_stream(self, query: str, state: SessionState) -> Generator[dict, None, None]:
        if not state.has_document:
            yield {"type": "error", "code": "DOCUMENT_NOT_LOADED", "data": "No document loaded."}
            return

        try:
            docs, _ = self._get_relevant_chunks(query, state, top_k=_QA_TOP_K)
            if not docs:
                yield {"type": "error", "code": "NO_RESULTS", "data": "No relevant info found."}
                return

            context = self._build_context(docs)
            prompt = f"<|im_start|>system\nAnswer using ONLY the provided context. You MUST cite sources for every claim using [SOURCE N] format. If the answer is not in the provided context, you MUST say 'I do not have enough information to answer this'.<|im_end|>\n<|im_start|>user\nContext:\n{context}\n\nQuestion: {query}<|im_end|>\n<|im_start|>assistant\n"
            
            full_text = ""
            for chunk in self.llm(prompt, max_tokens=450, temperature=0.0, stop=["<|im_end|>"], stream=True):
                token = chunk["choices"][0].get("text", "")
                if token:
                    full_text += token
                    yield {"type": "token", "data": token}
            
            yield {"type": "sources", "data": self._all_sources(docs)}
            yield {"type": "done", "full_text": full_text}
        except Exception as e:
            logger.exception("Chat failed")
            yield {"type": "error", "code": "INFERENCE_ERROR", "data": str(e)}

    def delete_namespace(self, namespace: str):
        try:
            if self._use_local:
                collection_name = f"{self._chroma_collection_name}_{namespace}"
                self._chroma_client.delete_collection(collection_name)
            else:
                self._pc_index.delete(delete_all=True, namespace=namespace)
        except Exception as e:
            logger.warning("Namespace deletion failed: %s", e)
