import streamlit as st
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.RAG.rag_pipeline import TOSAssistant

st.set_page_config(page_title="TOS Summarizer", layout="wide")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "qwen-merged-q4_k_m.gguf"

st.sidebar.title("Settings")
st.sidebar.info("Legal Document Summarizer powered by Qwen 2.5 (Fine-Tuned)")

@st.cache_resource
def load_rag_engine():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Please check your data folder.")
        return None
    return TOSAssistant(model_path=str(MODEL_PATH))

rag = load_rag_engine()

st.title("📜 Terms of Service Summarizer")
st.markdown("""
This tool uses a **Hybrid RAG approach**:
1. **Global Summarization:** Uses a fine-tuned model to summarize the entire document.
2. **Q&A:** Uses vector retrieval to find specific clauses for your questions.
""")

uploaded_file = st.file_uploader("Upload a Terms of Service (PDF)", type="pdf")

if uploaded_file:
    with open("temp_tos.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        with st.spinner("Ingesting document... (Chunking & Embedding)"):
            rag.ingest_document("temp_tos.pdf")
            st.session_state.last_uploaded = uploaded_file.name
            st.success("Document processed!")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Executive Summary")
        if st.button("Generate Summary"):
            with st.spinner("Reading full document and generating summary..."):
                summary = rag.generate_global_summary()
                st.text_area("Summary", value=summary, height=400)

    with col2:
        st.subheader("💬 Ask Questions")
        user_query = st.text_input("Ask about specific clauses (e.g., 'Can I get a refund?')")
        if st.button("Get Answer"):
            if user_query:
                with st.spinner("Searching document..."):
                    answer = rag.answer_question(user_query)
                    st.write(answer)
            else:
                st.warning("Please enter a question.")
else:
    st.info("Please upload a PDF to begin.")