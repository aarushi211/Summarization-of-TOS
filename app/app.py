import streamlit as st
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.RAG.rag2 import TOSAssistant

st.set_page_config(page_title="TOS Summarizer", layout="wide")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FINETUNED_MODEL_PATH = PROJECT_ROOT / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
BASE_MODEL_PATH = PROJECT_ROOT / "models" / "legal_qwen.Q4_K_M.gguf"

st.sidebar.title("Settings")
st.sidebar.info("Legal Document Summarizer powered by Qwen 2.5 (Fine-Tuned)")

@st.cache_resource
def load_rag_engine():
    return TOSAssistant(str(BASE_MODEL_PATH), str(FINETUNED_MODEL_PATH))

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

    # --- NEW: Metadata Inputs ---
    st.markdown("### 🏷️ Document Details")
    st.caption("Providing these details helps the AI generate a more accurate summary.")
    
    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        service_name = st.text_input(
            "Service Name", 
            value="YouTube", 
            placeholder="e.g. Spotify, Netflix, Google",
            help="The name of the company or service this document belongs to."
        )
    with meta_col2:
        doc_type = st.text_input(
            "Document Type", 
            value="Terms of Service", 
            placeholder="e.g. Privacy Policy, EULA",
            help="The specific type of legal document."
        )
    st.divider()
    # ----------------------------

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Executive Summary")
        if st.button("Generate Summary"):
            with st.spinner("Reading full document and generating summary..."):
                # Update the backend instance with user inputs before generating
                rag.service_name = service_name
                rag.doc_type = doc_type
                
                summary = rag.generate_global_summary()
                st.text_area("Summary", value=summary, height=600)

    with col2:
        st.subheader("💬 Ask Questions")
        user_query = st.text_input("Ask about specific clauses (e.g., 'Can I get a refund?')")
        if st.button("Get Answer"):
            if user_query:
                with st.spinner("Searching document..."):
                    answer = rag.answer_question(user_query)
                    
                    # Display Answer
                    st.markdown("### Answer")
                    st.write(answer["answer"])
                    
                    # Display Sources
                    with st.expander("View Source Clauses"):
                        for idx, source in enumerate(answer["sources"]):
                            st.markdown(f"**Source {idx+1}**")
                            st.caption(source) # Assuming sources is a list of strings
            else:
                st.warning("Please enter a question.")
else:
    st.info("Please upload a PDF to begin.")