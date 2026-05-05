import logging
import os
import time
import re
from pathlib import Path
from typing import Generator, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
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
from src.RAG.processors import clean_text, convert_to_markdown, sanitise_label
from src.RAG.loaders import load_pdf, load_text

logger = logging.getLogger(__name__)

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
        region: str = "us-east-1"
    ):
        self._metrics: list[dict] = []
        
        # Pinecone setup
        self._pc = PineconeClient(api_key=pinecone_api_key)
        self._index_name = index_name
        self._dimension = dimension
        self._ensure_pinecone_index(index_name, cloud, region)
        self._pc_index = self._pc.Index(index_name)
        
        # Load Models
        self.llm = Llama(model_path=model_path, n_ctx=8192, n_gpu_layers=-1, verbose=False)
        
        PROJECT_ROOT = Path(model_path).resolve().parent.parent
        local_embed = PROJECT_ROOT / "models" / "embeddings"
        local_cross = PROJECT_ROOT / "models" / "cross-encoder" / "ms-marco-MiniLM-L-6-v2"
        
        self.embed_model = HuggingFaceEmbeddings(model_name=str(local_embed))
        self.cross_encoder = CrossEncoder(str(local_cross))
        
        # Splitters
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "doc_title"), ("##", "section"), ("###", "subsection"), ("####", "clause")],
            strip_headers=False
        )
        self.sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3500, chunk_overlap=700, separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _ensure_pinecone_index(self, index_name, cloud, region):
        existing = {idx.name for idx in self._pc.list_indexes()}
        if index_name not in existing:
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
                time.sleep(2)
            raise EnvironmentError(f"Pinecone index {index_name} not ready.")

    def _build_chunks(self, documents: list[Document]) -> list[Document]:
        all_chunks: list[Document] = []
        chunk_id = 0
        for doc in documents:
            base_page = doc.metadata.get("page", 0)
            md_text = convert_to_markdown(clean_text(doc.page_content))
            sections = self.header_splitter.split_text(md_text)
            for section in sections:
                sub_chunks = self.sub_splitter.split_documents([section])
                for chunk in sub_chunks:
                    body = chunk.page_content
                    page_m = re.search(r'<!--\s*page:(\d+)\s*-->', body)
                    page_1idx = int(page_m.group(1)) if page_m else (base_page + 1)
                    body = re.sub(r'<!--.*?-->', '', body).strip()
                    if not body: continue
                    
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
    def _upsert_to_pinecone(self, chunks: list[Document], namespace: str):
        PineconeVectorStore.from_documents(
            chunks, self.embed_model, index_name=self._index_name,
            namespace=namespace
        )

    def _index_chunks(self, documents: list[Document], metrics: dict, state: SessionState):
        chunks = self._build_chunks(documents)
        state.cached_chunks = chunks
        metrics["chunk_count"] = len(chunks)
        if not chunks:
            raise IngestionError("Chunking produced zero chunks.")
        self._upsert_to_pinecone(chunks, state.pinecone_namespace)

    def ingest_document(self, pdf_path: str, state: SessionState):
        docs = load_pdf(pdf_path)
        state.full_text = '\n'.join(clean_text(d.page_content) for d in docs)
        self._index_chunks(docs, {"stage": "ingest"}, state)

    def ingest_text_file(self, txt_path: str, state: SessionState):
        docs = load_text(txt_path)
        state.full_text = '\n'.join(clean_text(d.page_content) for d in docs)
        self._index_chunks(docs, {"stage": "ingest"}, state)

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
    def _pinecone_search(self, query: str, namespace: str, k: int = 50) -> list[Document]:
        vec_store = PineconeVectorStore(
            index=self._pc_index, embedding=self.embed_model, namespace=namespace
        )
        return vec_store.similarity_search(query, k=k)

    @traceable(name="Hybrid Retrieval")
    def _get_relevant_chunks(self, query: str, state: SessionState, top_k: int = 7) -> tuple[list[Document], dict]:
        rm: dict = {}
        try:
            t0 = time.perf_counter()
            pinecone_results = self._pinecone_search(query, state.pinecone_namespace)
            rm["retrieval_s"] = time.perf_counter() - t0
        except RetryError as exc:
            raise RetrievalError("Vector search unavailable.") from exc

        if not pinecone_results: return [], rm

        bm25 = BM25Retriever.from_documents(pinecone_results, k=30)
        bm25_results = bm25.invoke(query)
        candidates = self._reciprocal_rank_fusion(pinecone_results, bm25_results)[:50]

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
            "Keep your summary concise (2-4 sentences)."
        )

        for label, query in SUMMARY_TOPICS:
            try:
                docs, _ = self._get_relevant_chunks(query, state, top_k=3)
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
            docs, _ = self._get_relevant_chunks(query, state, top_k=7)
            if not docs:
                yield {"type": "error", "code": "NO_RESULTS", "data": "No relevant info found."}
                return

            context = self._build_context(docs)
            prompt = f"<|im_start|>system\nAnswer using ONLY the provided context. You MUST cite sources for every claim using [SOURCE N] format.<|im_end|>\n<|im_start|>user\nContext:\n{context}\n\nQuestion: {query}<|im_end|>\n<|im_start|>assistant\n"
            
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
            self._pc_index.delete(delete_all=True, namespace=namespace)
        except Exception as e:
            logger.warning("Namespace deletion failed: %s", e)
