import logging
import os

# Suppress transformers logging
from transformers.utils import logging as transformers_logging
transformers_logging.set_verbosity_error()

# Optional: Suppress general Windows/Library warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from llama_cpp import Llama
from sentence_transformers import CrossEncoder
from pinecone import Pinecone as PineconeClient
import re
import time
import dataclasses
from langsmith import traceable


# ---------------------------------------------------------------------------
# Per-session document state (isolated per user/tab)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class SessionState:
    """
    Lightweight per-session state. No in-memory vector store.
    All embeddings live in Pinecone under `pinecone_namespace`.
    """
    pinecone_namespace: str  = ""      # "{user_id}_{document_id}" — unique per upload
    document_id:        str  = ""      # UUID of the row in the Supabase documents table
    service_name:       str  = "Unknown Service"
    doc_type:           str  = "Unknown Document"
    last_accessed:      float = dataclasses.field(default_factory=lambda: __import__('time').time())
    # Cached candidates for BM25 in-memory rerank (populated on first ingest)
    cached_chunks:      list = dataclasses.field(default_factory=list)

    @property
    def has_document(self) -> bool:
        return bool(self.pinecone_namespace)

    def reset(self):
        """Clear document state while keeping session alive."""
        self.pinecone_namespace = ""
        self.document_id        = ""
        self.service_name       = "Unknown Service"
        self.doc_type           = "Unknown Document"
        self.cached_chunks      = []


# ---------------------------------------------------------------------------
# Topics the RAG-based summariser retrieves chunks for.
# Each entry: (display_label, retrieval_query)
# ---------------------------------------------------------------------------
SUMMARY_TOPICS = [
    ("Data Collection",         "what personal data and information is collected from users"),
    ("Data Sharing / Selling",  "sharing selling transferring user data to third parties partners"),
    ("Data Retention & Deletion","data retention period deletion of user account and personal data"),
    ("User Rights",             "user rights account termination suspension appeals"),
    ("Refund & Cancellation",   "refund cancellation subscription termination policy"),
    ("Arbitration & Disputes",  "arbitration dispute resolution class action waiver"),
    ("Liability Limitations",   "limitation of liability damages indemnification"),
    ("IP & Content Ownership",  "intellectual property content ownership license user generated"),
    ("Policy Changes",          "changes to terms notice modification policy updates"),
    ("Governing Law",           "governing law jurisdiction venue applicable law"),
    ("Children & COPPA",        "children under 13 COPPA minors privacy"),
    ("Cookies & Tracking",      "cookies tracking pixels analytics advertising identifiers"),
]


class TOSAssistant:
    """
    Holds ALL shared, expensive ML model resources (LLM, embeddings, cross-encoder,
    splitters). These are loaded once at startup and shared across all user sessions.

    Per-session document state (vector store, BM25 index, chunks, etc.) lives in
    a SessionState object that is passed to each method, enabling true multi-tenancy
    without duplicating the heavy models.
    """

    def __init__(self, model_path, index_dir="faiss_index"):
        self.index_dir = Path(index_dir)
        self._metrics  = []

        # ── Pinecone client (shared, stateless connection) ───────────────────
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        index_name       = os.getenv("PINECONE_INDEX_NAME", "tos-summarizer")
        if not pinecone_api_key:
            raise EnvironmentError("PINECONE_API_KEY is not set in the environment.")
        self._pc          = PineconeClient(api_key=pinecone_api_key)
        self._pc_index    = self._pc.Index(index_name)
        self._index_name  = index_name
        print(f"Connected to Pinecone index: {index_name}")

        # ── GGUF Model (shared) ──────────────────────────────────────────────
        print(f"Loading RAG Model: {Path(model_path).name}")
        t0 = time.perf_counter()
        self.llm = Llama(
            model_path=model_path,
            n_ctx=8192,
            n_gpu_layers=-1,
            verbose=False,
        )
        self._metrics.append({
            "stage": "init", "sub_stage": "gguf_model_load",
            "latency_s": time.perf_counter() - t0,
        })

        SCRIPT_DIR   = Path(__file__).resolve().parent
        PROJECT_ROOT = SCRIPT_DIR.parent.parent
        local_embed  = PROJECT_ROOT / "models" / "embeddings"
        local_cross  = PROJECT_ROOT / "models" / "cross-encoder" / "ms-marco-MiniLM-L-6-v2"

        # ── Embedding Model (shared) ─────────────────────────────────────────
        print("Loading Embeddings & Cross-Encoder...")
        t0 = time.perf_counter()
        self.embed_model = HuggingFaceEmbeddings(
            model_name=str(local_embed),
            model_kwargs={"trust_remote_code": True},
        )
        self._metrics.append({
            "stage": "init", "sub_stage": "embedding_load",
            "latency_s": time.perf_counter() - t0,
        })

        # ── Cross-Encoder (shared) ───────────────────────────────────────────
        t0 = time.perf_counter()
        self.cross_encoder = CrossEncoder(str(local_cross))
        self._metrics.append({
            "stage": "init", "sub_stage": "cross_encoder_load",
            "latency_s": time.perf_counter() - t0,
        })

        # ── Splitters (shared, stateless — safe to reuse across sessions) ────
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#",    "doc_title"),
                ("##",   "section"),
                ("###",  "subsection"),
                ("####", "clause"),
            ],
            strip_headers=False,
        )
        self.sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3500,
            chunk_overlap=700,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        # NOTE: No per-session state here — all document state lives in SessionState.

    # =========================================================================
    # Pre-processing: raw text → structured Markdown
    # =========================================================================

    # Compiled header-detection patterns
    _RE_PAGE       = re.compile(r'^---\s*PAGE\s+(\d+)\s*---$', re.IGNORECASE)
    _RE_SOURCE_TAG = re.compile(r'<source_id>(.+?)</source_id>')
    _RE_DEEP       = re.compile(r'^(\d+\.\d+\.\d+)\s+([A-Z].+)$')              # 1.2.3 Header
    _RE_SUB        = re.compile(r'^(\d+\.\d+)\s+([A-Z][^\n]{2,})$')            # 1.2 Header
    _RE_TOP        = re.compile(r'^(\d+)\.\s+([A-Z][^\n]{3,})$')               # 1. Header
    _RE_LETTERED   = re.compile(r'^\(([a-z])\)\s+([A-Z].+)$')                  # (a) Sub-clause
    _RE_ARTICLE    = re.compile(r'^(Article\s+[IVXLCDM\d]+)[:\.\s]*(.*)',       re.IGNORECASE)
    _RE_SECTION_KW = re.compile(r'^(Section\s+\d+(?:\.\d+)*)[:\.\s]*(.*)',      re.IGNORECASE)
    _RE_APPENDIX   = re.compile(r'^(Appendix\s+[A-Z\d])[:\.\s]*(.*)',           re.IGNORECASE)
    _RE_SYMBOL     = re.compile(r'^(§\s*\d+(?:\.\d+)*)\s+(.*)')                # § 5 Disputes
    _RE_ALLCAPS    = re.compile(r'^([A-Z][A-Z\s\-\&\/]{4,})(?::|\.|\s*$)')     # ALL CAPS HEADER
    _RE_TITLE_CASE = re.compile(                                                # How We Use Your Data
        r'^((?:[A-Z][a-z]+\s){2,6}(?:[A-Z][a-z]+))(?::|$)'
    )

    def _convert_to_markdown(self, raw_text: str) -> str:
        """
        Convert raw TOS text into structured Markdown so MarkdownHeaderTextSplitter
        can split on true document boundaries.

        Rules
        -----
        - Page markers  → HTML comments <!-- page:N -->  (harvested post-split,
                          invisible to the header splitter)
        - Source tags   → preserved as-is, harvested post-split
        - Section lines → normalised to ##/###/#### Markdown headings
        - Everything else → left unchanged
        """
        output: list[str] = []

        for line in raw_text.split('\n'):
            stripped = line.strip()

            # Page markers ────────────────────────────────────────────────────
            m = self._RE_PAGE.match(stripped)
            if m:
                output.append(f'<!-- page:{m.group(1)} -->')
                continue

            # Source-ID tags (preserve, harvest later) ────────────────────────
            if self._RE_SOURCE_TAG.search(stripped):
                output.append(stripped)
                continue

            # Multi-level numeric: 1.2.3 ──────────────────────────────────────
            m = self._RE_DEEP.match(stripped)
            if m:
                output.append(f'#### {m.group(1)} {m.group(2)}')
                continue

            # Sub-section: 1.2 ────────────────────────────────────────────────
            m = self._RE_SUB.match(stripped)
            if m:
                output.append(f'### {m.group(1)} {m.group(2)}')
                continue

            # Top-level numeric: 1. ───────────────────────────────────────────
            m = self._RE_TOP.match(stripped)
            if m:
                output.append(f'## {m.group(1)}. {m.group(2)}')
                continue

            # Lettered sub-clause: (a) ────────────────────────────────────────
            m = self._RE_LETTERED.match(stripped)
            if m:
                output.append(f'#### ({m.group(1)}) {m.group(2)}')
                continue

            # Keyword-based: Article / Section / Appendix / § ─────────────────
            matched_keyword = False
            for pattern, md_level in [
                (self._RE_ARTICLE,    "##"),
                (self._RE_SECTION_KW, "###"),
                (self._RE_APPENDIX,   "##"),
                (self._RE_SYMBOL,     "###"),
            ]:
                m = pattern.match(stripped)
                if m:
                    rest    = m.group(2).strip()
                    heading = f"{m.group(1)}{': ' + rest if rest else ''}"
                    output.append(f'{md_level} {heading}')
                    matched_keyword = True
                    break

            if matched_keyword:
                continue

            # ALL-CAPS header (guard against all-caps paragraphs with len < 80) ─
            m = self._RE_ALLCAPS.match(stripped)
            if m and len(stripped) < 80:
                output.append(f'## {m.group(1).strip()}')
                continue

            # Title-Case prose header ─────────────────────────────────────────
            m = self._RE_TITLE_CASE.match(stripped)
            if m and len(stripped) < 80:
                output.append(f'### {m.group(1).strip()}')
                continue

            # Default: preserve original line ─────────────────────────────────
            output.append(line)

        return '\n'.join(output)

    # =========================================================================
    # Text cleaning (ligatures, whitespace normalisation)
    # =========================================================================

    _LIGATURE_MAP = {
        "\u00ef\u00ac\u0081": "fi",
        "\u00ef\u00ac\u0082": "fl",
        "\u00ef\u00ac\u0080": "ff",
        "\u00ef\u00ac\u0083": "ffi",
        "\u00ef\u00ac\u0084": "ffl",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "": "",
    }

    def clean_text(self, text: str) -> str:
        for bad, good in self._LIGATURE_MAP.items():
            text = text.replace(bad, good)
        text = re.sub(r'[ \t]+', ' ', text)        # collapse horizontal whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)     # max two consecutive newlines
        return text.strip()

    # =========================================================================
    # Two-stage chunking with full metadata injection
    # =========================================================================

    def _build_chunks(self, documents: list[Document]) -> list[Document]:
        """
        Pipeline per Document:
          1. Clean text
          2. Convert to Markdown (normalise headers, embed <!-- page:N --> comments)
          3. Stage-1: MarkdownHeaderTextSplitter  → section-scoped docs
          4. Stage-2: RecursiveCharacterTextSplitter → RAG-sized chunks
          5. Inject metadata: chunk_id, page, section, subsection, clause,
             source_id_first/last, citation string
        """
        all_chunks: list[Document] = []
        chunk_id = 0

        for doc in documents:
            base_page = doc.metadata.get("page", 0)   # 0-indexed (PyPDFLoader)
            md_text   = self._convert_to_markdown(self.clean_text(doc.page_content))

            # Stage 1 — structural split
            sections = self.header_splitter.split_text(md_text)

            for section in sections:
                # Stage 2 — size split within the section
                sub_chunks = self.sub_splitter.split_documents([section])

                for chunk in sub_chunks:
                    body = chunk.page_content

                    # Harvest page number from embedded comment
                    page_m    = re.search(r'<!--\s*page:(\d+)\s*-->', body)
                    page_1idx = int(page_m.group(1)) if page_m else (base_page + 1)
                    body      = re.sub(r'<!--.*?-->', '', body).strip()

                    # Harvest & strip source_id tags
                    source_ids = self._RE_SOURCE_TAG.findall(body)
                    body       = self._RE_SOURCE_TAG.sub('', body).strip()

                    chunk.page_content = body
                    if not body:
                        continue

                    # Pull propagated header metadata from Stage 1
                    section_title = chunk.metadata.get("section",    "General")
                    subsection    = chunk.metadata.get("subsection", "")
                    clause        = chunk.metadata.get("clause",     "")

                    # Build human-readable citation string
                    citation_parts = [f"p.{page_1idx}"]
                    if section_title and section_title != "General":
                        citation_parts.append(section_title)
                    if subsection:
                        citation_parts.append(subsection)
                    if clause:
                        citation_parts.append(clause)

                    chunk.metadata.update({
                        "chunk_id":        chunk_id,
                        "page":            page_1idx - 1,   # 0-indexed internally
                        "page_label":      f"p.{page_1idx}",
                        "section":         section_title,
                        "subsection":      subsection,
                        "clause":          clause,
                        "source_id_first": source_ids[0]  if source_ids else "",
                        "source_id_last":  source_ids[-1] if source_ids else "",
                        "citation":        " › ".join(citation_parts),
                    })

                    all_chunks.append(chunk)
                    chunk_id += 1

        return all_chunks

    # =========================================================================
    # Ingestion — PDF
    # =========================================================================

    def ingest_document(self, pdf_path: str, state: SessionState):
        print(f"Ingesting PDF: {pdf_path}")
        m: dict = {"stage": "ingest_document", "document": str(pdf_path)}

        t0 = time.perf_counter()
        documents = PyPDFLoader(pdf_path).load()
        m["pdf_parse_s"] = time.perf_counter() - t0
        m["page_count"]  = len(documents)

        t0 = time.perf_counter()
        state.full_text = '\n'.join(self.clean_text(d.page_content) for d in documents)
        m["text_clean_s"] = time.perf_counter() - t0
        m["total_chars"]  = len(state.full_text)

        self._index_chunks(documents, m, state)
        print("PDF ingestion complete.")

    # =========================================================================
    # Ingestion — plain text / scraped URL
    # =========================================================================

    def ingest_text_file(self, txt_path: str, state: SessionState):
        print(f"Ingesting text file: {txt_path}")
        m: dict = {"stage": "ingest_text", "document": str(txt_path)}

        t0 = time.perf_counter()
        documents = TextLoader(txt_path, encoding='utf-8').load()
        m["text_load_s"] = time.perf_counter() - t0

        for doc in documents:
            doc.metadata.setdefault("page", 0)   # no real pages in scraped text

        state.full_text  = self.clean_text(documents[0].page_content)
        m["total_chars"] = len(state.full_text)

        self._index_chunks(documents, m, state)
        print("Text ingestion complete.")

    # =========================================================================
    # Shared indexing logic
    # =========================================================================

    def _index_chunks(self, documents: list[Document], metrics: dict, state: SessionState):
        t0 = time.perf_counter()
        chunks              = self._build_chunks(documents)
        state.cached_chunks = chunks          # keep for BM25 reranking
        metrics["chunking_s"]  = time.perf_counter() - t0
        metrics["chunk_count"] = len(chunks)

        if not chunks:
            print("WARNING: No chunks produced — check document content.")
            return

        # Embed and upsert to Pinecone (namespaced per user+document)
        t0 = time.perf_counter()
        PineconeVectorStore.from_documents(
            chunks,
            self.embed_model,
            index_name=self._index_name,
            namespace=state.pinecone_namespace,
        )
        metrics["pinecone_upsert_s"] = time.perf_counter() - t0

        metrics["total_ingest_s"] = (
            metrics.get("pdf_parse_s", 0) +
            metrics.get("text_load_s", 0) +
            metrics.get("text_clean_s", 0) +
            metrics["chunking_s"] +
            metrics["pinecone_upsert_s"]
        )
        self._metrics.append(metrics)

    # =========================================================================
    # Retrieval — Hybrid BM25 + FAISS MMR → RRF → Cross-Encoder rerank
    # =========================================================================

    @staticmethod
    def _reciprocal_rank_fusion(*ranked_lists, k: int = 60) -> list[Document]:
        scores:  dict[int, float]    = {}
        doc_map: dict[int, Document] = {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                cid          = doc.metadata.get("chunk_id", id(doc))
                scores[cid]  = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
                doc_map[cid] = doc
        return [doc_map[cid] for cid in sorted(scores, key=scores.__getitem__, reverse=True)]

    @traceable(name="Hybrid Retrieval (Pinecone + BM25)")
    def _get_relevant_chunks(
        self, query: str, state: SessionState, top_k: int = 7
    ) -> tuple[list[Document], dict]:
        rm: dict = {}

        # Step 1: Pinecone semantic search (namespaced to this user's document)
        t0 = time.perf_counter()
        vec_store     = PineconeVectorStore(
            index=self._pc_index,
            embedding=self.embed_model,
            namespace=state.pinecone_namespace,
        )
        pinecone_results = vec_store.similarity_search(query, k=50)
        rm["pinecone_search_s"]   = time.perf_counter() - t0
        rm["pinecone_candidates"] = len(pinecone_results)

        if not pinecone_results:
            return [], rm

        # Step 2: Inline BM25 rerank on the Pinecone candidate set
        # (no need to index all chunks — BM25 here scores only the 50 candidates)
        t0 = time.perf_counter()
        bm25 = BM25Retriever.from_documents(pinecone_results, k=30)
        bm25_results = bm25.invoke(query)
        rm["bm25_rerank_s"]    = time.perf_counter() - t0
        rm["bm25_candidates"]  = len(bm25_results)

        # Step 3: Merge Pinecone + BM25 with RRF then cross-encode top-50
        candidates = self._reciprocal_rank_fusion(pinecone_results, bm25_results)[:50]

        # Step 4: Cross-encoder rerank for precision (uses shared model)
        t0 = time.perf_counter()
        pairs  = [[query, doc.page_content] for doc in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        rm["rerank_s"] = time.perf_counter() - t0

        rm["mmr_search_s"] = rm["pinecone_search_s"]  # keep key name for metrics compat
        rm["total_retrieval_s"] = rm["pinecone_search_s"] + rm["bm25_rerank_s"] + rm["rerank_s"]
        return [doc for doc, _ in ranked[:top_k]], rm

    # =========================================================================
    # LLM helpers
    # =========================================================================

    def format_qwen_prompt(self, system_msg: str, user_msg: str) -> str:
        return (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    @traceable(name="Llama.cpp Inference")
    def _llm_chat(
        self, system_msg: str, user_msg: str, max_tokens: int = 400
    ) -> tuple[str, dict]:
        """Chat-completion wrapper; returns (answer_text, usage_metrics)."""
        t0     = time.perf_counter()
        output = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            repeat_penalty=1.1,
            stop=["<|im_end|>"],
        )
        elapsed = time.perf_counter() - t0
        usage   = output.get("usage", {})
        comp    = usage.get("completion_tokens", 0)
        return output['choices'][0]['message']['content'], {
            "llm_inference_s":   elapsed,
            "prompt_tokens":     usage.get("prompt_tokens", 0),
            "completion_tokens": comp,
            "total_tokens":      usage.get("total_tokens", 0),
            "tokens_per_sec":    comp / elapsed if elapsed > 0 and comp > 0 else 0,
        }

    # =========================================================================
    # RAG-based summarisation with per-topic citations
    # =========================================================================

    @traceable(name="Generate Global Summary")
    def generate_global_summary(self, state: SessionState) -> dict:
        """
        For every topic in SUMMARY_TOPICS:
          1. Retrieve the top-3 most relevant chunks via hybrid search.
          2. Ask the LLM to write 2-4 sentences about the topic, with [SOURCE N] tags.
          3. Parse which sources were actually cited.

        Returns:
            {
              "topics": [
                  {
                    "label":   "Data Collection",
                    "summary": "The service collects ... [SOURCE 1].",
                    "sources": [ {tag, citation, section, page, excerpt}, ... ]
                  },
                  ...
              ],
              "metrics": { ... }
            }
        """
        if not state.has_document:
            return {"topics": [], "error": "No document loaded."}

        metrics_entry: dict = {"stage": "global_summary", "topics": []}
        topic_results: list = []
        seen_ids: set[int]  = set()   # deduplicate chunks across topics
        service_name        = state.service_name
        doc_type            = state.doc_type
        t_total = time.perf_counter()

        system_msg = (
            f"You are a legal expert summarising a {doc_type} for {service_name}. "
            "Using ONLY the provided source excerpts, write 2-4 concise sentences on the topic. "
            "After each factual claim add a citation tag like [SOURCE N]. "
            "If the topic is not covered in the sources, write exactly: 'NOT_IN_DOCUMENT'."
        )

        for label, query in SUMMARY_TOPICS:
            t0   = time.perf_counter()
            docs, rm = self._get_relevant_chunks(query, state, top_k=3)

            # Prefer fresh chunks; fall back to all retrieved if every chunk is a duplicate
            fresh = [d for d in docs if d.metadata["chunk_id"] not in seen_ids] or docs

            context_parts = []
            for i, doc in enumerate(fresh):
                seen_ids.add(doc.metadata["chunk_id"])
                context_parts.append(
                    f"[SOURCE {i+1} | {doc.metadata.get('citation', '')}]:\n{doc.page_content}"
                )

            user_msg     = f"Topic: {label}\n\nSources:\n" + "\n\n".join(context_parts)
            summary_text, llm_m = self._llm_chat(system_msg, user_msg, max_tokens=250)

            # Parse cited indices
            cited_indices = {
                int(m.group(1)) - 1
                for m in re.finditer(r'\[SOURCE (\d+)\]', summary_text)
                if 0 <= int(m.group(1)) - 1 < len(fresh)
            }
            cited_sources = [
                {
                    "tag":      f"[SOURCE {idx+1}]",
                    "citation": fresh[idx].metadata.get("citation", ""),
                    "section":  fresh[idx].metadata.get("section", "General"),
                    "page":     fresh[idx].metadata.get("page", 0) + 1,
                    "excerpt":  fresh[idx].page_content[:350] + "...",
                }
                for idx in sorted(cited_indices)
            ]

            topic_results.append({
                "label":   label,
                "summary": summary_text,
                "sources": cited_sources,
            })
            metrics_entry["topics"].append({
                "label":             label,
                "retrieval_s":       rm["total_retrieval_s"],
                "llm_inference_s":   llm_m["llm_inference_s"],
                "total_s":           time.perf_counter() - t0,
                "prompt_tokens":     llm_m["prompt_tokens"],
                "completion_tokens": llm_m["completion_tokens"],
            })

        metrics_entry["total_summary_s"] = time.perf_counter() - t_total
        self._metrics.append(metrics_entry)
        return {"topics": topic_results, "metrics": metrics_entry}

    # =========================================================================
    # QA with inline citations
    # =========================================================================

    @traceable(name="QA Inference")
    def answer_question(self, query: str, state: SessionState) -> dict:
        if not state.has_document:
            return {
                "answer": "Please ingest a document first.",
                "cited_sources": [], "all_retrieved": [],
            }

        metrics_entry: dict = {"stage": "qa_inference", "query": query[:80]}

        # Retrieval
        relevant_docs, rm = self._get_relevant_chunks(query, state, top_k=7)
        metrics_entry.update(rm)

        # Context assembly
        context_parts: list[str] = []
        for i, doc in enumerate(relevant_docs):
            citation   = doc.metadata.get("citation", f"chunk-{i}")
            section    = doc.metadata.get("section", "")
            subsection = doc.metadata.get("subsection", "")
            header     = f"[SOURCE {i+1} | {citation}]"
            if section and section != "General":
                header += f"\nSection: {section}"
                if subsection:
                    header += f" › {subsection}"
            context_parts.append(f"{header}:\n{doc.page_content}")

        context_str = "\n\n".join(context_parts)
        metrics_entry["context_chars"] = len(context_str)

        system_msg = (
            "You are a strict legal assistant. Answer the user's question using ONLY the provided "
            "Context sources. After EVERY factual claim cite its source using [SOURCE N]. "
            "Multiple sources per sentence are allowed: [SOURCE 1][SOURCE 3]. "
            "Do not infer 'selling' of data unless the text explicitly states 'we sell data'. "
            "Distinguish 'sharing' (for functionality) from 'selling' (for profit). "
            "If the answer is absent from all sources, respond: 'NOT_IN_DOCUMENT'."
        )
        user_msg = f"Context:\n{context_str}\n\nQuestion: {query}"

        answer_text, llm_m = self._llm_chat(system_msg, user_msg, max_tokens=450)
        metrics_entry.update(llm_m)
        metrics_entry["total_qa_s"] = rm["total_retrieval_s"] + llm_m["llm_inference_s"]
        self._metrics.append(metrics_entry)

        # Parse [SOURCE N] references that appear in the answer
        cited_indices = {
            int(m.group(1)) - 1
            for m in re.finditer(r'\[SOURCE (\d+)\]', answer_text)
            if 0 <= int(m.group(1)) - 1 < len(relevant_docs)
        }

        def _source_dict(idx: int, doc: Document) -> dict:
            return {
                "tag":        f"[SOURCE {idx+1}]",
                "citation":   doc.metadata.get("citation", ""),
                "section":    doc.metadata.get("section", "General"),
                "subsection": doc.metadata.get("subsection", ""),
                "page":       doc.metadata.get("page", 0) + 1,
                "excerpt":    doc.page_content[:400] + (
                    "..." if len(doc.page_content) > 400 else ""
                ),
            }

        return {
            "answer": answer_text,
            "cited_sources": [
                _source_dict(idx, relevant_docs[idx]) for idx in sorted(cited_indices)
            ],
            "all_retrieved": [
                {
                    "tag":      f"[SOURCE {i+1}]",
                    "citation": d.metadata.get("citation", ""),
                    "section":  d.metadata.get("section", "General"),
                    "page":     d.metadata.get("page", 0) + 1,
                    "excerpt":  d.page_content[:200] + "...",
                }
                for i, d in enumerate(relevant_docs)
            ],
        }

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_metrics(self) -> list:
        return list(self._metrics)

    def reset_metrics(self):
        self._metrics = [m for m in self._metrics if m.get("stage") == "init"]