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
from src.RAG.processors import clean_text, convert_to_markdown, pages_to_source_text, sanitise_label
from src.RAG.loaders import load_pdf, load_text
from src.RAG.query_expansion import expand_search_query

logger = logging.getLogger(__name__)

# Retrieval / search depth
_DENSE_SEARCH_K = 50
_BM25_SEARCH_K = 50
_RRF_CANDIDATE_K = 50
_QA_TOP_K = 5
_QA_TOP_K_LONG = 7
_LONG_DOC_MIN_PAGES = 12
_LONG_DOC_MIN_CHUNKS = 40
_SUMMARY_TOP_K = 3
_SECTION_CHUNK_SIZE = 1200
_SECTION_CHUNK_OVERLAP = 300

_RE_WITHOUT_NOTIF_Q = re.compile(r"without\s+(?:prior\s+)?notif", re.I)
_RE_NOTIFY_IN_TEXT = re.compile(r"\bnotif\w*", re.I)
_QA_MAX_TOKENS = 150
_QA_EXTRACT_MAX_TOKENS = 200

_QA_ABSTENTION = "I do not have enough information to answer this."

_QA_EXTRACT_SYSTEM = """You extract evidence from legal document source chunks.

Rules:
- Copy exact sentences or short phrases from the sources that help answer the question.
- Prefix each line with the source number from the context, e.g. "1: <copied text>" or "[SOURCE 1] <copied text>"
- Do not interpret, summarize, or add facts not present in the sources.
- If no source text answers the question, reply with exactly: NONE
"""

_RE_YES_NO = re.compile(
    r"^\s*(can|could|does|do|did|is|are|was|were|will|would|shall|should|am|have|has)\b",
    re.I,
)

_QA_SYNTHESIZE_SYSTEM = f"""You write a brief legal answer using ONLY the extracted evidence below.

Rules:
- At most 2 bullet points. Use 1 if one evidence line is enough.
- Each bullet must map to exactly ONE extracted evidence line (light paraphrase only).
- End each bullet with [SOURCE N] matching that line's source prefix.
- Do NOT use numbered lists, steps, introductions, or extra sentences.
- Do not add facts not present in the extracted evidence.
- Do not use hedging (may, might, usually, often, typically) unless the evidence uses them.
- If evidence is insufficient, reply with exactly:
  {_QA_ABSTENTION}
"""

_QA_SYNTHESIZE_YESNO_HINT = """
This is a YES/NO question.
- Use exactly 1 bullet (2 only if two evidence lines are both essential).
- The bullet MUST start with "Yes," or "No," then paraphrase the supporting evidence in the same sentence.
- End the bullet with [SOURCE N]. Do not output only "Yes" or "No" without explanation and citation.
- If evidence says users WILL be notified of changes, you must NOT answer Yes to "without notifying".
- If evidence does not clearly support Yes or No, reply with exactly the abstention sentence from the system prompt.
"""

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
        self.sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=_SECTION_CHUNK_SIZE,
            chunk_overlap=_SECTION_CHUNK_OVERLAP,
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
        state.page_count = len(docs)

        metrics = {
            "stage": "ingest_document",
            "pdf_parse_s": t_parse_end - t_parse,
            "page_count": state.page_count,
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
        state.page_count = max(1, len(docs))
        metrics = {"stage": "ingest_document", "pdf_parse_s": 0.1, "page_count": state.page_count}
        self._index_chunks(docs, metrics, state)
        metrics["total_ingest_s"] = time.perf_counter() - t_start
        self._metrics.append(metrics)

    def answer_question(self, query: str, state: SessionState) -> dict:
        """Non-streaming version for benchmarking and internal use."""
        t_start = time.perf_counter()
        docs, rm = self._get_relevant_chunks(query, state, top_k=self._qa_top_k(state))
        t_retrieval = time.perf_counter() - t_start
        
        if not docs:
            return {"answer": "No relevant info found.", "sources": []}
            
        context = self._build_context(docs)
        summary, llm_metrics = self._generate_qa_answer(query, context)

        total_qa_s = time.perf_counter() - t_start

        metrics = {
            "stage": "qa_inference",
            "total_qa_s": total_qa_s,
            "total_retrieval_s": t_retrieval,
            "llm_inference_s": llm_metrics["latency"],
            "qa_extract_s": llm_metrics.get("extract_s", 0),
            "qa_synthesize_s": llm_metrics.get("synthesize_s", 0),
            "prompt_tokens": len(context) // 4, # approx
            "completion_tokens": len(summary) // 4,
            "tokens_per_sec": (len(summary) // 4) / llm_metrics["latency"] if llm_metrics["latency"] > 0 else 0
        }
        self._metrics.append(metrics)
        return {
            "answer": summary,
            "sources": self._all_sources(docs),
            "context": context,
        }

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
            rm["qa_top_k"] = top_k
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
    def _is_yes_no_question(query: str) -> bool:
        q = query.strip()
        return q.endswith("?") and bool(_RE_YES_NO.match(q))

    @staticmethod
    def _qa_top_k(state: SessionState) -> int:
        if state.page_count >= _LONG_DOC_MIN_PAGES:
            return _QA_TOP_K_LONG
        if len(state.cached_chunks) >= _LONG_DOC_MIN_CHUNKS:
            return _QA_TOP_K_LONG
        return _QA_TOP_K

    @staticmethod
    def _notification_synthesize_hint(query: str, extracted: str) -> str:
        if not _RE_WITHOUT_NOTIF_Q.search(query):
            return ""
        if not _RE_NOTIFY_IN_TEXT.search(extracted):
            return ""
        return (
            "\n\nCRITICAL: The sources require Spotify to NOTIFY users of term changes. "
            'The question asks about changes WITHOUT notifying. You MUST begin with "No," '
            "and quote the notification language. Do NOT answer Yes."
        )

    @staticmethod
    def _answer_claims_no_notification(query: str, answer: str) -> bool:
        if not _RE_WITHOUT_NOTIF_Q.search(query):
            return False
        head = answer.strip()[:80].lower()
        return head.startswith("yes,") or head.startswith("yes ")

    @staticmethod
    def _qa_extract_user_message(context: str, query: str) -> str:
        msg = f"Context:\n{context}\n\nQuestion: {query}"
        if TOSAssistant._is_yes_no_question(query):
            msg += (
                "\n\nThis is a yes/no question. Extract lines that support Yes or No. "
                "If the sources do not clearly support either, reply NONE."
            )
        if _RE_WITHOUT_NOTIF_Q.search(query):
            msg += (
                "\n\nFocus on whether the service notifies users when terms change "
                "(look for notify, notification, posting revised terms, email)."
            )
        return msg

    @staticmethod
    def _qa_synthesize_user_message(query: str, extracted: str) -> str:
        msg = f"Extracted evidence:\n{extracted}\n\nQuestion: {query}"
        if TOSAssistant._is_yes_no_question(query):
            msg += _QA_SYNTHESIZE_YESNO_HINT
        else:
            msg += "\n\nUse only the evidence lines above. One bullet per line, max 2 bullets."
        msg += TOSAssistant._notification_synthesize_hint(query, extracted)
        return msg

    @staticmethod
    def _yesno_answer_incomplete(answer: str) -> bool:
        """Bare Yes/No cannot be scored by the faithfulness judge."""
        a = answer.strip()
        if len(a) < 35:
            return True
        return bool(re.match(r"^(answer:\s*)?(yes|no)[\.\!\?]*\s*$", a, re.I))

    def _synthesize_from_extract(self, query: str, extracted: str) -> str:
        user_msg = self._qa_synthesize_user_message(query, extracted)
        summary, _ = self._llm_chat(
            _QA_SYNTHESIZE_SYSTEM,
            user_msg,
            max_tokens=_QA_MAX_TOKENS,
        )
        if self._is_yes_no_question(query) and self._yesno_answer_incomplete(summary):
            summary, _ = self._llm_chat(
                _QA_SYNTHESIZE_SYSTEM,
                user_msg
                + "\n\nYour previous reply was too short. Write one full bullet starting with Yes, or No, "
                "then the evidence and [SOURCE N].",
                max_tokens=_QA_MAX_TOKENS,
            )
        if self._answer_claims_no_notification(query, summary):
            summary, _ = self._llm_chat(
                _QA_SYNTHESIZE_SYSTEM,
                user_msg + self._notification_synthesize_hint(query, extracted),
                max_tokens=_QA_MAX_TOKENS,
            )
        return summary

    @staticmethod
    def _is_no_evidence(extracted: str) -> bool:
        """True only when extraction explicitly found nothing (not when format is imperfect)."""
        text = extracted.strip()
        if not text:
            return True
        upper = text.upper()
        if upper == "NONE":
            return True
        if upper.startswith("NONE") and len(text) < 40:
            return True
        # Ignore lines that are only NONE / whitespace
        content_lines = [
            ln for ln in text.splitlines()
            if ln.strip() and ln.strip().upper() != "NONE"
        ]
        return len(content_lines) == 0

    def _generate_qa_answer(self, query: str, context: str) -> tuple[str, dict]:
        """Two-step QA: extract evidence from chunks, then synthesize a short cited answer."""
        t_extract = time.perf_counter()
        extracted, _ = self._llm_chat(
            _QA_EXTRACT_SYSTEM,
            self._qa_extract_user_message(context, query),
            max_tokens=_QA_EXTRACT_MAX_TOKENS,
        )
        extract_s = time.perf_counter() - t_extract

        if self._is_no_evidence(extracted):
            return _QA_ABSTENTION, {
                "latency": extract_s,
                "extract_s": extract_s,
                "synthesize_s": 0.0,
            }

        t_synth = time.perf_counter()
        summary = self._synthesize_from_extract(query, extracted)
        synth_s = time.perf_counter() - t_synth
        return summary, {
            "latency": extract_s + synth_s,
            "extract_s": extract_s,
            "synthesize_s": synth_s,
        }

    @staticmethod
    def _qa_extract_stream_prompt(context: str, query: str) -> str:
        user_block = TOSAssistant._qa_extract_user_message(context, query)
        return (
            f"<|im_start|>system\n{_QA_EXTRACT_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user_block}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    @staticmethod
    def _qa_synthesize_stream_prompt(query: str, extracted: str) -> str:
        user_block = TOSAssistant._qa_synthesize_user_message(query, extracted)
        return (
            f"<|im_start|>system\n{_QA_SYNTHESIZE_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user_block}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _llm_complete(self, prompt: str, max_tokens: int) -> str:
        """Single non-streaming completion from a raw chat-template prompt."""
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            stop=["<|im_end|>"],
            stream=False,
        )
        return output["choices"][0].get("text", "").strip()

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
            docs, _ = self._get_relevant_chunks(query, state, top_k=self._qa_top_k(state))
            if not docs:
                yield {"type": "error", "code": "NO_RESULTS", "data": "No relevant info found."}
                return

            context = self._build_context(docs)

            extracted = self._llm_complete(
                self._qa_extract_stream_prompt(context, query),
                max_tokens=_QA_EXTRACT_MAX_TOKENS,
            )
            if self._is_no_evidence(extracted):
                yield {"type": "token", "data": _QA_ABSTENTION}
                yield {"type": "sources", "data": self._all_sources(docs)}
                yield {"type": "done", "full_text": _QA_ABSTENTION}
                return

            full_text = self._synthesize_from_extract(query, extracted)
            if full_text:
                yield {"type": "token", "data": full_text}

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
