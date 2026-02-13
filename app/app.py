# import streamlit as st
# import os
# import sys
# from pathlib import Path

# # --- SETUP PATHS ---
# # Ensure we can find the src module regardless of where this script is run
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from src.RAG.rag_pipeline import TOSAssistant

# # --- PAGE CONFIG ---
# st.set_page_config(
#     page_title="TOS Summarizer",
#     page_icon="📜",
#     layout="wide"
# )

# # --- CONSTANTS ---
# SCRIPT_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = SCRIPT_DIR.parent
# # Ensure this matches your actual model path
# MODEL_PATH = PROJECT_ROOT / "models" / "legal_qwen.Q4_K_M.gguf"

# # --- SIDEBAR INFO ---
# st.sidebar.title("ℹ️ About")
# st.sidebar.info("Legal Document Summarizer powered by Qwen 2.5 (Fine-Tuned)")
# st.sidebar.markdown("### ⚠️ Disclaimer")
# st.sidebar.warning(
#     """
#     **Not Legal Advice.** This tool uses Artificial Intelligence to summarize legal documents. 
#     AI models can make mistakes ("hallucinate") and may not capture every nuance of a contract.
    
#     * **Do not rely** on this summary for legal decisions.
#     * **Always verify** important clauses in the original PDF.
#     * This tool is for informational purposes only.
#     """
# )

# # --- LOAD RAG ENGINE ---
# @st.cache_resource
# def load_rag_engine():
#     # Only load the model once
#     if not MODEL_PATH.exists():
#         st.error(f"Model not found at: {MODEL_PATH}")
#         st.stop()
#     return TOSAssistant(str(MODEL_PATH))

# rag = load_rag_engine()

# # --- MAIN TITLE ---
# st.title("📜 Terms of Service Summarizer")
# st.markdown("""
# This tool uses a **Hybrid RAG approach**:
# 1. **Global Summarization:** Uses a fine-tuned model to summarize the entire document.
# 2. **Q&A:** Uses vector retrieval to find specific clauses for your questions.
# """)

# st.divider()

# # --- FILE UPLOAD ---
# uploaded_file = st.file_uploader("Upload a Terms of Service (PDF)", type="pdf")

# if uploaded_file:
#     # Save file temporarily
#     with open("temp_tos.pdf", "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     # Process file only if it's new
#     if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
#         with st.spinner("Ingesting document... (Chunking & Embedding)"):
#             rag.ingest_document("temp_tos.pdf")
#             st.session_state.last_uploaded = uploaded_file.name
#             st.success(f"✅ Processed: {uploaded_file.name}")

#     st.markdown("### 🏷️ Document Details")
#     st.caption("Providing these details helps the AI generate a more accurate summary.")
    
#     meta_col1, meta_col2 = st.columns(2)
#     with meta_col1:
#         service_name = st.text_input(
#             "Service Name", 
#             value=st.session_state.get("service_name", ""), 
#             placeholder="e.g. Spotify, Netflix, Google",
#             help="The name of the company or service this document belongs to."
#         )
#         st.session_state.service_name = service_name
        
#     with meta_col2:
#         doc_type = st.text_input(
#             "Document Type", 
#             value=st.session_state.get("doc_type", ""), 
#             placeholder="e.g. Privacy Policy, EULA",
#             help="The specific type of legal document."
#         )
#         st.session_state.doc_type = doc_type

#     st.divider()

#     # --- TWO COLUMN INTERFACE ---
#     col_summary, col_qa = st.columns([1, 1])

#     # --- LEFT COLUMN: SUMMARY ---
#     with col_summary:
#         st.subheader("📝 Executive Summary")
        
#         if st.button("Generate Global Summary", use_container_width=True):
#             if not rag.full_text:
#                 st.error("Please ingest a document first.")
#             else:
#                 with st.spinner("Reading full document and generating summary..."):
#                     # Update metadata in RAG
#                     rag.service_name = service_name if service_name else "Unknown Service"
#                     rag.doc_type = doc_type if doc_type else "Legal Document"
                    
#                     summary_text = rag.generate_global_summary()
#                     # Store summary in session state so it persists
#                     st.session_state.summary = summary_text
        
#         # Display summary if it exists in state
#         if "summary" in st.session_state:
#             st.text_area("Summary Result", value=st.session_state.summary, height=600)

#     # --- RIGHT COLUMN: Q&A ---
#     with col_qa:
#         st.subheader("💬 Ask Questions")
        
#         # Initialize chat history
#         if "messages" not in st.session_state:
#             st.session_state.messages = []

#         # Display chat history
#         for message in st.session_state.messages:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])
#                 if "sources" in message:
#                     with st.expander("View Sources"):
#                         for idx, src in enumerate(message["sources"]):
#                             st.caption(f"**Source {idx+1}:** {src}")

#         # Chat Input
#         if prompt := st.chat_input("Ask about specific clauses (e.g., 'Can I get a refund?')"):
#             # Add user message to history
#             st.session_state.messages.append({"role": "user", "content": prompt})
#             with st.chat_message("user"):
#                 st.markdown(prompt)

#             # Generate Answer
#             with st.chat_message("assistant"):
#                 with st.spinner("Searching document..."):
#                     rag.service_name = service_name
#                     # Get answer from RAG pipeline
#                     response = rag.answer_question(prompt)
                    
#                     st.markdown(response["answer"])
                    
#                     # Show sources in expander
#                     with st.expander("View Source Clauses"):
#                         for idx, source in enumerate(response["sources"]):
#                             st.markdown(f"**Source {idx+1}**")
#                             st.caption(source)

#             # Add assistant response to history
#             st.session_state.messages.append({
#                 "role": "assistant", 
#                 "content": response["answer"],
#                 "sources": response["sources"]
#             })

#     # --- BOTTOM SECTION: RESET BUTTON ---
#     st.divider()
#     st.markdown("### 🔄 Start Over")
#     st.caption("Finished with this document? Click below to clear everything and upload a new file.")
    
#     if st.button("🗑️ Clear History & Reset", type="primary", use_container_width=True):
#         # Clear RAG memory
#         rag.full_text = ""
#         rag.vector_store = None
#         rag.service_name = "Unknown Service"
#         rag.doc_type = "Unknown Doc Type"
        
#         # Clear Streamlit session state
#         st.session_state.clear()
        
#         # Remove temp file if it exists
#         if os.path.exists("temp_tos.pdf"):
#             os.remove("temp_tos.pdf")
            
#         st.rerun()

# else:
#     # Empty State (When no file is uploaded)
#     st.info("👋 Welcome! Please upload a PDF file to begin analyzing legal documents.")

# import streamlit as st
# import os
# import sys
# import requests
# from bs4 import BeautifulSoup
# from pathlib import Path

# # --- SETUP PATHS ---
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from src.RAG.rag_pipeline import TOSAssistant

# # --- PAGE CONFIG ---
# st.set_page_config(page_title="TOS Summarizer", page_icon="📜", layout="wide")

# SCRIPT_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = SCRIPT_DIR.parent
# MODEL_PATH = PROJECT_ROOT / "models" / "legal_qwen.Q4_K_M.gguf"

# # --- HELPER: URL SCRAPER ---
# def scrape_tos_from_url(url):
#     """Scrapes text from a webpage, cleaning out scripts and styles."""
#     try:
#         headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
#         response = requests.get(url, headers=headers, timeout=10)
        
#         if response.status_code != 200:
#             return None, f"Failed to retrieve page (Status Code: {response.status_code})"
        
#         soup = BeautifulSoup(response.content, 'html.parser')
        
#         # Kill all script and style elements
#         for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
#             script.extract()
            
#         # Get text
#         text = soup.get_text(separator=' ')
        
#         # Break into lines and remove leading/trailing space on each
#         lines = (line.strip() for line in text.splitlines())
#         # Drop blank lines
#         text = '\n'.join(line for line in lines if line)
        
#         return text, None
#     except Exception as e:
#         return None, str(e)

# # --- LOAD RAG ENGINE ---
# @st.cache_resource
# def load_rag_engine():
#     if not MODEL_PATH.exists():
#         st.error(f"Model not found at: {MODEL_PATH}")
#         st.stop()
#     return TOSAssistant(str(MODEL_PATH))

# rag = load_rag_engine()

# # --- MAIN UI ---
# st.title("📜 Terms of Service Summarizer")
# st.markdown("Summarize legal documents using **Local RAG** (Qwen 2.5 1.5B).")

# # --- INPUT METHOD SELECTION ---
# tab1, tab2 = st.tabs(["📄 Upload PDF", "🔗 Paste URL"])

# with tab1:
#     uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
#     if uploaded_file:
#         # Check if we already processed this file to avoid re-ingesting
#         if "current_source" not in st.session_state or st.session_state.current_source != uploaded_file.name:
#             with open("temp_tos.pdf", "wb") as f:
#                 f.write(uploaded_file.getbuffer())
            
#             with st.spinner("Processing PDF..."):
#                 rag.ingest_document("temp_tos.pdf")
#                 st.session_state.current_source = uploaded_file.name
#                 st.session_state.source_type = "pdf"
#                 st.success("PDF processed successfully!")

# with tab2:
#     url_input = st.text_input("Enter the direct link to Terms of Service page")
    
#     # Helper link to find TOS easily
#     if not url_input:
#         st.caption("Don't have the link? [Search Google for 'Terms of Service'](https://www.google.com/search?q=terms+of+service)")

#     if st.button("Scrape & Analyze URL"):
#         if url_input:
#             if "current_source" not in st.session_state or st.session_state.current_source != url_input:
#                 with st.spinner(f"Scraping content from {url_input}..."):
#                     text_content, error = scrape_tos_from_url(url_input)
                    
#                     if error:
#                         st.error(f"Error scraping URL: {error}")
#                     elif len(text_content) < 500:
#                         st.warning("Warning: Scraped content is very short. This might not be a valid TOS page.")
#                     else:
#                         # Save as temp text file for ingestion
#                         with open("temp_webpage.txt", "w", encoding="utf-8") as f:
#                             f.write(text_content)
                        
#                         # Ingest the text file
#                         rag.ingest_text_file("temp_webpage.txt") # You need to add this method to your RAG class
                        
#                         st.session_state.current_source = url_input
#                         st.session_state.source_type = "url"
#                         st.success("Webpage scraped and processed!")

# st.divider()

# # --- DOCUMENT DETAILS (Metadata) ---
# # Only show the rest of the UI if we have a loaded document
# if "current_source" in st.session_state:
#     st.markdown(f"**Analyzing:** `{st.session_state.current_source}`")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         service_name = st.text_input("Service Name", value=st.session_state.get("service_name", ""), placeholder="e.g. Netflix")
#         st.session_state.service_name = service_name
#     with col2:
#         doc_type = st.text_input("Document Type", value=st.session_state.get("doc_type", ""), placeholder="e.g. Privacy Policy")
#         st.session_state.doc_type = doc_type

#     # --- SUMMARY & Q&A SECTION ---
#     col_sum, col_qa = st.columns(2)
    
#     with col_sum:
#         st.subheader("📝 Summary")
#         if st.button("Generate Summary"):
#             rag.service_name = service_name
#             rag.doc_type = doc_type
#             with st.spinner("Generating summary..."):
#                 summary = rag.generate_global_summary()
#                 st.session_state.summary = summary
        
#         if "summary" in st.session_state:
#             st.text_area("Global Summary", st.session_state.summary, height=500)

#     with col_qa:
#         st.subheader("💬 Q&A")
#         # Chat history logic (Same as before)
#         if "messages" not in st.session_state:
#             st.session_state.messages = []

#         for msg in st.session_state.messages:
#             st.chat_message(msg["role"]).write(msg["content"])

#         if prompt := st.chat_input("Ask about the document..."):
#             st.session_state.messages.append({"role": "user", "content": prompt})
#             st.chat_message("user").write(prompt)
            
#             with st.spinner("Thinking..."):
#                 rag.service_name = service_name
#                 response = rag.answer_question(prompt)
#                 st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
#                 st.chat_message("assistant").write(response["answer"])
                
#                 with st.expander("Sources"):
#                     for s in response["sources"]:
#                         st.write(s)

#     # --- RESET ---
#     st.divider()
#     if st.button("Start Over"):
#         st.session_state.clear()
#         st.rerun()

# else:
#     st.info("Please Upload a PDF or Enter a URL to begin.")

import streamlit as st
import os
import sys
import requests
from bs4 import BeautifulSoup
from pathlib import Path

# --- SETUP PATHS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.RAG.rag_pipeline import TOSAssistant

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TOS Summarizer",
    page_icon="📜",
    layout="wide"
)

# --- CONSTANTS ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "legal_qwen.Q4_K_M.gguf"

# --- SIDEBAR INFO & DISCLAIMER ---
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

# --- HELPER: URL SCRAPER ---
def scrape_tos_from_url(url):
    """Scrapes text from a webpage, cleaning out scripts and styles."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None, f"Failed to retrieve page (Status Code: {response.status_code})"
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Kill all script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.extract()
            
        # Get text
        text = soup.get_text(separator=' ')
        
        # Break into lines and remove leading/trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Drop blank lines
        text = '\n'.join(line for line in lines if line)
        
        return text, None
    except Exception as e:
        return None, str(e)

# --- LOAD RAG ENGINE ---
@st.cache_resource
def load_rag_engine():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at: {MODEL_PATH}")
        st.stop()
    return TOSAssistant(str(MODEL_PATH))

rag = load_rag_engine()

# --- MAIN TITLE & EXPLANATION ---
st.title("📜 Terms of Service Summarizer")
st.markdown("""
Welcome! This tool helps you quickly understand complex legal documents like **Terms of Service**, **Privacy Policies**, and **EULAs**. 

Instead of reading 50 pages of legal jargon, you can:
1.  **Upload a PDF** or **Paste a Link** to the contract.
2.  Get an **Instant Summary** of your rights and obligations.
3.  **Chat with the Document** to ask specific questions (e.g., *"Can I get a refund?"*).
""")

st.divider()

# --- INPUT METHOD SELECTION ---
tab1, tab2 = st.tabs(["📄 Upload PDF", "🔗 Paste URL"])

with tab1:
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file:
        # Check if we already processed this file to avoid re-ingesting
        if "current_source" not in st.session_state or st.session_state.current_source != uploaded_file.name:
            # Save file temporarily
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
                        # Save as temp text file for ingestion
                        with open("temp_webpage.txt", "w", encoding="utf-8") as f:
                            f.write(text_content)
                        
                        # Ingest the text file
                        rag.ingest_text_file("temp_webpage.txt") 
                        
                        st.session_state.current_source = url_input
                        st.session_state.source_type = "url"
                        st.success("✅ Webpage scraped and processed!")

st.divider()

# --- DOCUMENT DETAILS (Metadata) ---
# Only show the rest of the UI if we have a loaded document
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

    # --- SUMMARY & Q&A SECTION ---
    col_sum, col_qa = st.columns([1, 1])
    
    # --- LEFT: SUMMARY ---
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

    # --- RIGHT: Q&A ---
    with col_qa:
        st.subheader("💬 Ask Questions")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "sources" in msg:
                    with st.expander("View Sources"):
                        for idx, s in enumerate(msg["sources"]):
                            st.caption(f"**Source {idx+1}:** {s}")

        # Chat Input
        if prompt := st.chat_input("Ask about the document... (e.g. 'Can I sue them?')"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            with st.spinner("Thinking..."):
                rag.service_name = service_name
                response = rag.answer_question(prompt)
                
                # Display Answer
                st.chat_message("assistant").write(response["answer"])
                
                # Show Sources
                with st.expander("View Source Clauses"):
                    for idx, s in enumerate(response["sources"]):
                        st.markdown(f"**Source {idx+1}**")
                        st.caption(s)

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response["answer"],
                    "sources": response["sources"]
                })

    # --- RESET ---
    st.divider()
    st.markdown("### 🔄 Start Over")
    st.caption("Finished with this document? Click below to clear everything and analyze a new one.")
    
    if st.button("🗑️ Clear History & Reset", type="primary"):
        # Clear RAG memory
        rag.full_text = ""
        rag.vector_store = None
        rag.service_name = "Unknown Service"
        rag.doc_type = "Unknown Doc Type"
        
        # Clear Streamlit session state
        st.session_state.clear()
        
        # Cleanup temp files
        if os.path.exists("temp_tos.pdf"): os.remove("temp_tos.pdf")
        if os.path.exists("temp_webpage.txt"): os.remove("temp_webpage.txt")
            
        st.rerun()

else:
    # Empty State Hint
    st.info("👆 **Get Started:** Upload a PDF or Paste a URL in the tabs above to begin analysis.")