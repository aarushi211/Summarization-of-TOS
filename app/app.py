import streamlit as st
import os
import sys
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.RAG.rag_pipeline import TOSAssistant

st.set_page_config(
    page_title="TOS Summarizer",
    page_icon="📜",
    layout="wide",
)

IS_CLOUD = os.getenv("CLOUD_RUN_ENV", "False") == "True"
if IS_CLOUD:
    PROJECT_ROOT = Path("/app")
else:
    SCRIPT_DIR   = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent

MODEL_PATH = PROJECT_ROOT / "models" / "legal_qwen.Q4_K_M.gguf"

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("ℹ️ About")
st.sidebar.info("Legal Document Summarizer powered by Qwen 2.5 (Fine-Tuned)")

st.sidebar.markdown("### ⚠️ Disclaimer")
st.sidebar.warning(
    "**Not Legal Advice.** This tool uses AI to summarize legal documents. "
    "AI models can make mistakes and may not capture every nuance.\n\n"
    "* **Do not rely** on this for legal decisions.\n"
    "* **Always verify** important clauses in the original document.\n"
    "* For informational purposes only."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ How it Works")
st.sidebar.caption(
    "**Hybrid RAG Pipeline:**\n"
    "1. **Markdown pre-processing** — normalises section headers so the splitter "
    "understands document structure.\n"
    "2. **Two-stage chunking** — MarkdownHeaderSplitter (structure) → "
    "RecursiveCharacterSplitter (size).\n"
    "3. **BM25 + FAISS + RRF + Cross-Encoder** — hybrid retrieval with reranking.\n"
    "4. **RAG summarisation** — 12 legal topics retrieved independently; "
    "every claim is cited back to a page & section."
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def scrape_tos_from_url(url: str) -> tuple[str | None, str | None]:
    try:
        headers  = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}"
        soup = BeautifulSoup(response.content, 'html.parser')
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.extract()
        text  = soup.get_text(separator=' ')
        lines = (l.strip() for l in text.splitlines())
        return '\n'.join(l for l in lines if l), None
    except Exception as e:
        return None, str(e)


def render_cited_text(text: str, sources: list[dict]) -> str:
    """
    Replace [SOURCE N] tags with blue superscript footnote markers.
    Hovering over a marker shows the citation string as a tooltip.
    Returns HTML safe for st.markdown(unsafe_allow_html=True).
    """
    tag_to_fn: dict[str, int] = {s["tag"]: i + 1 for i, s in enumerate(sources)}

    def _replace(m: re.Match) -> str:
        tag = m.group(0)
        fn  = tag_to_fn.get(tag)
        if fn is None:
            return tag
        tooltip = sources[fn - 1]["citation"]
        return (
            f'<sup title="{tooltip}" style="color:#3B82F6;cursor:help;'
            f'font-weight:700;border-bottom:1px dotted #3B82F6;">[{fn}]</sup>'
        )

    return re.sub(r'\[SOURCE \d+\]', _replace, text)


def show_source_expanders(sources: list[dict], prefix: str = ""):
    """Render a numbered expander for each cited source."""
    for i, src in enumerate(sources):
        page_str    = f"Page {src['page']}" if src.get("page") else ""
        section_str = src.get("section", "")
        parts       = [p for p in [page_str, section_str] if p]
        label       = " · ".join(parts) or src.get("citation", f"Source {i+1}")
        with st.expander(f"{prefix}[{i+1}] {label}"):
            st.caption(f"**Citation:** {src.get('citation', '')}")
            if src.get("subsection"):
                st.caption(f"**Subsection:** {src['subsection']}")
            st.markdown(f"> {src.get('excerpt', '')}")


@st.cache_resource
def load_rag_engine() -> TOSAssistant:
    if not MODEL_PATH.exists():
        st.error(f"Model not found at: {MODEL_PATH}")
        st.stop()
    return TOSAssistant(str(MODEL_PATH))


rag = load_rag_engine()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("📜 Terms of Service Summarizer")
st.markdown(
    "Upload a PDF or paste a URL to get an **instant structured summary** "
    "and **cited Q&A** for any Terms of Service, Privacy Policy, or EULA."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────────────────────────────────────
tab_pdf, tab_url = st.tabs(["📄 Upload PDF", "🔗 Paste URL"])

with tab_pdf:
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file:
        if st.session_state.get("current_source") != uploaded_file.name:
            with open("temp_tos.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("Ingesting PDF — chunking, embedding, building indexes…"):
                rag.ingest_document("temp_tos.pdf")
            st.session_state.current_source = uploaded_file.name
            st.session_state.source_type    = "pdf"
            st.session_state.messages       = []
            st.session_state.pop("summary_data", None)
            st.success(f"✅ Processed: {uploaded_file.name}")

with tab_url:
    st.info("Best for static pages. If scraping fails, download the page as a PDF.")
    url_input = st.text_input("Enter the direct link to a TOS / Privacy Policy page")

    if st.button("Scrape & Analyze URL") and url_input:
        if st.session_state.get("current_source") != url_input:
            with st.spinner(f"Scraping {url_input}…"):
                text_content, error = scrape_tos_from_url(url_input)

            if error:
                st.error(f"Scraping error: {error}")
            elif not text_content or len(text_content) < 500:
                st.warning("Scraped content is very short — may not be a valid TOS page.")
            else:
                with open("temp_webpage.txt", "w", encoding="utf-8") as f:
                    f.write(text_content)
                with st.spinner("Embedding & indexing…"):
                    rag.ingest_text_file("temp_webpage.txt")
                st.session_state.current_source = url_input
                st.session_state.source_type    = "url"
                st.session_state.messages       = []
                st.session_state.pop("summary_data", None)
                st.success("✅ Webpage scraped and processed!")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Main UI — only after ingestion
# ─────────────────────────────────────────────────────────────────────────────
if "current_source" not in st.session_state:
    st.info("👆 **Get Started:** Upload a PDF or paste a URL above.")
    st.stop()

# Document details
st.markdown("### 🏷️ Document Details")
st.caption("These help the AI generate a more accurate summary.")
col1, col2 = st.columns(2)
with col1:
    service_name = st.text_input(
        "Service Name",
        value=st.session_state.get("service_name", ""),
        placeholder="e.g. Netflix, Spotify",
    )
    st.session_state.service_name = service_name
with col2:
    doc_type = st.text_input(
        "Document Type",
        value=st.session_state.get("doc_type", ""),
        placeholder="e.g. Privacy Policy, Terms of Use",
    )
    st.session_state.doc_type = doc_type

st.divider()

col_sum, col_qa = st.columns([1, 1])

# ─────────────────────────────────────────────────────────────────────────────
# Summary column — per-topic with citations
# ─────────────────────────────────────────────────────────────────────────────
with col_sum:
    st.subheader("📝 Executive Summary")
    st.caption(
        "Retrieves the most relevant clauses for 12 legal topics independently. "
        "Every sentence is cited back to its source page and section."
    )

    if st.button("Generate Summary", use_container_width=True):
        rag.service_name = service_name
        rag.doc_type     = doc_type
        with st.spinner(
            "Retrieving clauses for 12 legal topics and generating cited summary…"
        ):
            summary_data = rag.generate_global_summary()
        st.session_state.summary_data = summary_data

    if "summary_data" in st.session_state:
        data   = st.session_state.summary_data
        topics = data.get("topics", [])

        if not topics:
            st.warning(data.get("error", "No summary generated."))
        else:
            for topic in topics:
                label       = topic["label"]
                summary_txt = topic["summary"]
                sources     = topic["sources"]

                with st.expander(f"**{label}**", expanded=True):
                    if summary_txt.strip() == "NOT_IN_DOCUMENT":
                        st.caption("_Not covered in this document._")
                    elif sources:
                        rendered = render_cited_text(summary_txt, sources)
                        st.markdown(rendered, unsafe_allow_html=True)
                        st.markdown("---")
                        show_source_expanders(sources, prefix="  ")
                    else:
                        # Model answered but didn't produce [SOURCE N] tags
                        st.write(summary_txt)

# ─────────────────────────────────────────────────────────────────────────────
# QA column — cited chat
# ─────────────────────────────────────────────────────────────────────────────
with col_qa:
    st.subheader("💬 Ask Questions")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Render chat history ───────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                cited = msg.get("cited_sources", [])
                if cited:
                    rendered = render_cited_text(msg["content"], cited)
                    st.markdown(rendered, unsafe_allow_html=True)
                    st.markdown("---")
                    show_source_expanders(cited)
                else:
                    st.write(msg["content"])

                # Debug: all retrieved chunks (collapsed)
                if msg.get("all_retrieved"):
                    with st.expander("🔍 All Retrieved Chunks", expanded=False):
                        for chunk in msg["all_retrieved"]:
                            st.caption(
                                f"{chunk['tag']} · {chunk['citation']} · "
                                f"Section: {chunk['section']}"
                            )
                            st.markdown(f"> {chunk['excerpt']}")
                            st.markdown("---")
            else:
                st.write(msg["content"])

    # ── New user message ──────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about the document… (e.g. 'Can they sell my data?')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("Retrieving relevant clauses and generating answer…"):
            rag.service_name = service_name
            response         = rag.answer_question(prompt)

        answer_text  = response["answer"]
        cited        = response["cited_sources"]
        all_retrieved = response["all_retrieved"]

        with st.chat_message("assistant"):
            if cited:
                rendered = render_cited_text(answer_text, cited)
                st.markdown(rendered, unsafe_allow_html=True)
                st.markdown("---")
                show_source_expanders(cited)
            else:
                st.write(answer_text)

            if all_retrieved:
                with st.expander("🔍 All Retrieved Chunks", expanded=False):
                    for chunk in all_retrieved:
                        st.caption(
                            f"{chunk['tag']} · {chunk['citation']} · "
                            f"Section: {chunk['section']}"
                        )
                        st.markdown(f"> {chunk['excerpt']}")
                        st.markdown("---")

        st.session_state.messages.append({
            "role":          "assistant",
            "content":       answer_text,
            "cited_sources": cited,
            "all_retrieved": all_retrieved,
        })

# ─────────────────────────────────────────────────────────────────────────────
# Reset
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 🔄 Start Over")
st.caption("Clear everything and analyze a new document.")
if st.button("🗑️ Clear History & Reset", type="primary"):
    rag.full_text      = ""
    rag.vector_store   = None
    rag.bm25_retriever = None
    rag.all_chunks     = []
    rag.service_name   = "Unknown Service"
    rag.doc_type       = "Unknown Document"
    st.session_state.clear()
    for tmp in ["temp_tos.pdf", "temp_webpage.txt"]:
        if os.path.exists(tmp):
            os.remove(tmp)
    st.rerun()