import streamlit as st
import os
import sys
import requests
from bs4 import BeautifulSoup
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.RAG.rag_pipeline import TOSAssistant

st.set_page_config(
    page_title="TOS Summarizer",
    page_icon="📜",
    layout="wide"
)

PROJECT_ROOT = Path("/app")
MODEL_PATH = PROJECT_ROOT / "models" / "legal_qwen.Q4_K_M.gguf"

st.sidebar.title("ℹ️ About")
st.sidebar.info("Legal Document Summarizer powered by Qwen 2.5 (Fine-Tuned)")

st.sidebar.markdown("### ⚠️ Disclaimer")
st.sidebar.warning(
    """
    **Not Legal Advice.** This tool uses Artificial Intelligence to summarize legal documents. 
    AI models can make mistakes ("hallucinate") and may not capture every nuance of a contract.
    
    * **Do not rely** on this summary for legal decisions.
    * **Always verify** important clauses in the original PDF.
    * This tool is for informational purposes only.
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ How it Works")
st.sidebar.caption(
    "This tool uses a **Hybrid RAG approach**:\n"
    "1. **Global Summarization:** Summarizes the full document context.\n"
    "2. **Vector Retrieval:** Finds specific clauses to answer your questions."
)

def scrape_tos_from_url(url):
    """Scrapes text from a webpage, cleaning out scripts and styles."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None, f"Failed to retrieve page (Status Code: {response.status_code})"
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.extract()
            
        text = soup.get_text(separator=' ')
        
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)
        
        return text, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_rag_engine():
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"MODEL_PATH exists: {MODEL_PATH.exists()}")
    
    if not MODEL_PATH.exists():
        st.error(f"Model not found at: {MODEL_PATH}")
        st.stop()
    
    return TOSAssistant(str(MODEL_PATH))

rag = load_rag_engine()

st.title("📜 Terms of Service Summarizer")
st.markdown("""
Welcome! This tool helps you quickly understand complex legal documents like **Terms of Service**, **Privacy Policies**, and **EULAs**. 

Instead of reading 50 pages of legal jargon, you can:
1.  **Upload a PDF** or **Paste a Link** to the contract.
2.  Get an **Instant Summary** of your rights and obligations.
3.  **Chat with the Document** to ask specific questions (e.g., *"Can I get a refund?"*).
""")

st.divider()

tab1, tab2 = st.tabs(["📄 Upload PDF", "🔗 Paste URL"])

with tab1:
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file:
        if "current_source" not in st.session_state or st.session_state.current_source != uploaded_file.name:
            with open("temp_tos.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Ingesting document... (Chunking & Embedding)"):
                rag.ingest_document("temp_tos.pdf")
                st.session_state.current_source = uploaded_file.name
                st.session_state.source_type = "pdf"
                st.success(f"✅ Processed PDF: {uploaded_file.name}")

with tab2:
    url_input = st.text_input("Enter the direct link to Terms of Service page")
    
    if not url_input:
        st.caption("Don't have the link? [Search Google for 'Terms of Service'](https://www.google.com/search?q=terms+of+service)")

    if st.button("Scrape & Analyze URL"):
        if url_input:
            if "current_source" not in st.session_state or st.session_state.current_source != url_input:
                with st.spinner(f"Scraping content from {url_input}..."):
                    text_content, error = scrape_tos_from_url(url_input)
                    
                    if error:
                        st.error(f"Error scraping URL: {error}")
                    elif len(text_content) < 500:
                        st.warning("Warning: Scraped content is very short. This might not be a valid TOS page.")
                    else:
                        with open("temp_webpage.txt", "w", encoding="utf-8") as f:
                            f.write(text_content)
                        
                        rag.ingest_text_file("temp_webpage.txt") 
                        
                        st.session_state.current_source = url_input
                        st.session_state.source_type = "url"
                        st.success("✅ Webpage scraped and processed!")

st.divider()

if "current_source" in st.session_state:
    st.markdown(f"### 🏷️ Document Details")
    st.caption("Providing these details helps the AI generate a more accurate summary.")
    
    col1, col2 = st.columns(2)
    with col1:
        service_name = st.text_input(
            "Service Name", 
            value=st.session_state.get("service_name", ""), 
            placeholder="e.g. Netflix, Spotify",
            help="The name of the company or service."
        )
        st.session_state.service_name = service_name
    with col2:
        doc_type = st.text_input(
            "Document Type", 
            value=st.session_state.get("doc_type", ""), 
            placeholder="e.g. Privacy Policy, Terms of Use",
            help="The specific type of document."
        )
        st.session_state.doc_type = doc_type

    st.divider()

    col_sum, col_qa = st.columns([1, 1])
    
    with col_sum:
        st.subheader("📝 Executive Summary")
        if st.button("Generate Summary", use_container_width=True):
            rag.service_name = service_name
            rag.doc_type = doc_type
            with st.spinner("Reading document and generating summary..."):
                summary = rag.generate_global_summary()
                st.session_state.summary = summary
        
        if "summary" in st.session_state:
            st.text_area("Global Summary", st.session_state.summary, height=600)

    with col_qa:
        st.subheader("💬 Ask Questions")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "sources" in msg:
                    with st.expander("View Sources"):
                        for idx, s in enumerate(msg["sources"]):
                            st.caption(f"**Source {idx+1}:** {s}")

        if prompt := st.chat_input("Ask about the document... (e.g. 'Can I sue them?')"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            with st.spinner("Thinking..."):
                rag.service_name = service_name
                response = rag.answer_question(prompt)
                
                st.chat_message("assistant").write(response["answer"])
                
                with st.expander("View Source Clauses"):
                    for idx, s in enumerate(response["sources"]):
                        st.markdown(f"**Source {idx+1}**")
                        st.caption(s)

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response["answer"],
                    "sources": response["sources"]
                })

    st.divider()
    st.markdown("### 🔄 Start Over")
    st.caption("Finished with this document? Click below to clear everything and analyze a new one.")
    
    if st.button("🗑️ Clear History & Reset", type="primary"):
        rag.full_text = ""
        rag.vector_store = None
        rag.service_name = "Unknown Service"
        rag.doc_type = "Unknown Doc Type"
        
        st.session_state.clear()
        
        if os.path.exists("temp_tos.pdf"): os.remove("temp_tos.pdf")
        if os.path.exists("temp_webpage.txt"): os.remove("temp_webpage.txt")
            
        st.rerun()

else:
    st.info("👆 **Get Started:** Upload a PDF or Paste a URL in the tabs above to begin analysis.")
