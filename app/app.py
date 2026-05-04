import streamlit as st
import os
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="TOS Summarizer",
    page_icon="📜",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Auth helpers
# ─────────────────────────────────────────────────────────────────────────────

def _auth_headers() -> dict:
    token = st.session_state.get("access_token", "")
    return {"Authorization": f"Bearer {token}"}

def _is_logged_in() -> bool:
    return bool(st.session_state.get("access_token"))


def show_login_page():
    """Full-screen login / signup page shown when the user is not authenticated."""
    st.markdown(
        "<h1 style='text-align:center;margin-top:60px'>📜 TOS Summarizer</h1>"
        "<p style='text-align:center;color:gray'>AI-powered legal document analysis</p>",
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_signup = st.tabs(["🔐 Log In", "✏️ Sign Up"])

        with tab_login:
            email    = st.text_input("Email",    key="login_email")
            password = st.text_input("Password", key="login_password", type="password")
            if st.button("Log In", use_container_width=True, type="primary"):
                if not email or not password:
                    st.error("Please fill in both fields.")
                else:
                    with st.spinner("Logging in..."):
                        try:
                            resp = requests.post(
                                f"{API_URL}/auth/login",
                                json={"email": email, "password": password},
                                timeout=15,
                            )
                            if resp.status_code == 200:
                                data = resp.json()
                                st.session_state.access_token  = data["access_token"]
                                st.session_state.refresh_token = data["refresh_token"]
                                st.session_state.user_id       = data["user_id"]
                                st.session_state.user_email    = data["email"]
                                st.rerun()
                            else:
                                st.error(resp.json().get("detail", "Login failed."))
                        except Exception as e:
                            st.error(f"Connection error: {e}")

        with tab_signup:
            email    = st.text_input("Email",            key="signup_email")
            password = st.text_input("Password (min 6)", key="signup_password", type="password")
            if st.button("Create Account", use_container_width=True, type="primary"):
                if not email or not password:
                    st.error("Please fill in both fields.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    with st.spinner("Creating account..."):
                        try:
                            resp = requests.post(
                                f"{API_URL}/auth/signup",
                                json={"email": email, "password": password},
                                timeout=15,
                            )
                            if resp.status_code == 200:
                                st.success("Account created! Please check your email to confirm, then log in.")
                            else:
                                st.error(resp.json().get("detail", "Signup failed."))
                        except Exception as e:
                            st.error(f"Connection error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Gate: show login page if not authenticated
# ─────────────────────────────────────────────────────────────────────────────
if not _is_logged_in():
    show_login_page()
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (only reachable when logged in)
# ─────────────────────────────────────────────────────────────────────────────

def scrape_tos_from_url(url: str) -> tuple[str | None, str | None]:
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}"
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.extract()
        text  = soup.get_text(separator=" ")
        lines = (l.strip() for l in text.splitlines())
        return "\n".join(l for l in lines if l), None
    except Exception as e:
        return None, str(e)


def render_cited_text(text: str, sources: list[dict]) -> str:
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


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — user info + past documents
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"👤 **{st.session_state.get('user_email', 'User')}**")
    if st.button("🚪 Logout", use_container_width=True):
        for key in ["access_token", "refresh_token", "user_id", "user_email",
                    "current_doc_id", "current_source", "messages", "summary_data"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.markdown("---")
    st.markdown("### 📂 My Documents")

    # Fetch document history
    try:
        hist_resp = requests.get(f"{API_URL}/history/documents", headers=_auth_headers(), timeout=10)
        past_docs = hist_resp.json().get("documents", []) if hist_resp.status_code == 200 else []
    except Exception:
        past_docs = []

    if past_docs:
        for doc in past_docs:
            label     = f"📄 {doc['filename']}"
            sub_label = f"*{doc.get('service_name', '')} · {doc.get('doc_type', '')}*"
            with st.expander(label):
                st.caption(sub_label)
                if st.button("Load this document", key=f"load_{doc['id']}"):
                    st.session_state.current_doc_id  = doc["id"]
                    st.session_state.current_source  = doc["filename"]
                    st.session_state.source_type     = "history"
                    st.session_state.messages        = []
                    st.session_state.pop("summary_data", None)
                    # Load chat history for this doc
                    try:
                        chat_resp = requests.get(
                            f"{API_URL}/history/chats/{doc['id']}",
                            headers=_auth_headers(), timeout=10
                        )
                        sessions = chat_resp.json().get("sessions", []) if chat_resp.status_code == 200 else []
                        loaded_msgs = []
                        for sess in sessions:
                            for msg in sess.get("messages", []):
                                loaded_msgs.append({
                                    "role":           msg["role"],
                                    "content":        msg["content"],
                                    "cited_sources":  msg.get("cited_sources", []),
                                    "all_retrieved":  [],
                                })
                        st.session_state.messages = loaded_msgs
                    except Exception:
                        pass
                    st.rerun()
    else:
        st.caption("No documents yet. Upload one to get started!")

    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.warning(
        "**Not Legal Advice.** AI models can make mistakes.\n\n"
        "* **Do not rely** on this for legal decisions.\n"
        "* **Always verify** important clauses.\n"
        "* For informational purposes only."
    )
    st.markdown("### 🛠️ How it Works")
    st.caption(
        "**Hybrid RAG Pipeline:**\n"
        "1. Markdown pre-processing → structured headers.\n"
        "2. Two-stage chunking → Pinecone upsert.\n"
        "3. Pinecone search → BM25 rerank → Cross-Encoder.\n"
        "4. 12 legal topics retrieved & cited independently."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("📜 TOS Summarizer")
st.caption("Upload a Terms of Service or Privacy Policy. Get a cited AI summary and ask questions.")

# ─────────────────────────────────────────────────────────────────────────────
# Document input tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_pdf, tab_url = st.tabs(["📄 Upload PDF", "🔗 Scrape URL"])

with tab_pdf:
    service_name_pdf = st.text_input("Service name (e.g. Spotify)", key="svc_pdf")
    doc_type_pdf     = st.selectbox("Document type", ["Terms of Service", "Privacy Policy", "Cookie Policy", "EULA", "Other"], key="dt_pdf")
    uploaded_file    = st.file_uploader("Upload a TOS / Privacy Policy PDF", type=["pdf"])

    if uploaded_file and st.session_state.get("current_source") != uploaded_file.name:
        with open("temp_tos.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Uploading to S3 & ingesting into Pinecone…"):
            with open("temp_tos.pdf", "rb") as f:
                try:
                    resp = requests.post(
                        f"{API_URL}/ingest/pdf",
                        files={"file": (uploaded_file.name, f, "application/pdf")},
                        data={"service_name": service_name_pdf or "Unknown", "doc_type": doc_type_pdf},
                        headers=_auth_headers(),
                        timeout=120,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    st.session_state.current_doc_id  = result["document_id"]
                    st.session_state.current_source  = uploaded_file.name
                    st.session_state.source_type     = "pdf"
                    st.session_state.messages        = []
                    st.session_state.pop("summary_data", None)
                    st.success(f"✅ Ingested: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Ingest failed: {e}")
                    st.stop()

with tab_url:
    st.info("Best for static pages. If scraping fails, download as PDF.")
    service_name_url = st.text_input("Service name", key="svc_url")
    doc_type_url     = st.selectbox("Document type", ["Terms of Service", "Privacy Policy", "Cookie Policy", "EULA", "Other"], key="dt_url")
    url_input        = st.text_input("Enter the direct link to a TOS / Privacy Policy page")

    if st.button("Scrape & Analyze URL") and url_input:
        if st.session_state.get("current_source") != url_input:
            with st.spinner(f"Scraping {url_input}…"):
                text_content, error = scrape_tos_from_url(url_input)
            if error:
                st.error(f"Scraping error: {error}")
            elif not text_content or len(text_content) < 500:
                st.warning("Scraped content is very short — may not be a valid TOS page.")
            else:
                with st.spinner("Uploading & indexing via API…"):
                    try:
                        resp = requests.post(
                            f"{API_URL}/ingest/text",
                            data={
                                "text":         text_content,
                                "filename":     "scraped_page.txt",
                                "service_name": service_name_url or "Unknown",
                                "doc_type":     doc_type_url,
                            },
                            headers=_auth_headers(),
                            timeout=120,
                        )
                        resp.raise_for_status()
                        result = resp.json()
                        st.session_state.current_doc_id  = result["document_id"]
                        st.session_state.current_source  = url_input
                        st.session_state.source_type     = "url"
                        st.session_state.messages        = []
                        st.session_state.pop("summary_data", None)
                        st.success("✅ Webpage scraped and indexed!")
                    except Exception as e:
                        st.error(f"Ingest failed: {e}")
                        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Guard: nothing loaded yet
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.get("current_doc_id"):
    st.info("👆 Upload a PDF or paste a URL above to get started.")
    st.stop()

doc_id       = st.session_state.current_doc_id
service_name = st.session_state.get("service_name", "Unknown Service")
doc_type     = st.session_state.get("doc_type",     "Terms of Service")
st.success(f"✅ Active document: **{st.session_state.get('current_source', 'Unknown')}**")

# ─────────────────────────────────────────────────────────────────────────────
# Summary tab
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🔍 AI-Generated Summary")
st.caption("Covers 12 key legal topics. Every claim is cited back to a specific section.")

if st.button("Generate Summary", use_container_width=True):
    with st.spinner("Retrieving clauses for 12 legal topics and generating cited summary…"):
        try:
            resp = requests.post(
                f"{API_URL}/summary",
                json={"document_id": doc_id, "service_name": service_name, "doc_type": doc_type},
                headers=_auth_headers(),
                timeout=300,
            )
            resp.raise_for_status()
            st.session_state.summary_data = resp.json()
        except Exception as e:
            st.session_state.summary_data = {"error": f"API Error: {e}"}

if "summary_data" in st.session_state:
    summary_data = st.session_state.summary_data
    if "error" in summary_data:
        st.error(summary_data["error"])
    else:
        for topic in summary_data.get("topics", []):
            label   = topic.get("label", "")
            text    = topic.get("summary", "")
            sources = topic.get("sources", [])

            if "NOT_IN_DOCUMENT" in text:
                with st.expander(f"📌 {label}  *(not found in document)*"):
                    st.caption("This topic does not appear to be covered.")
                continue

            with st.expander(f"📌 {label}", expanded=False):
                st.markdown(
                    render_cited_text(text, sources),
                    unsafe_allow_html=True,
                )
                if sources:
                    st.markdown("**Sources:**")
                    show_source_expanders(sources, prefix="  ")

# ─────────────────────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 💬 Ask Questions")
st.caption("Ask anything about this document. The AI will answer using only the document text.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            cited   = msg.get("cited_sources", [])
            content = render_cited_text(msg["content"], cited) if cited else msg["content"]
            st.markdown(content, unsafe_allow_html=True)
            if cited:
                show_source_expanders(cited, prefix="  ")
        else:
            st.write(msg["content"])

if prompt := st.chat_input("Ask a question about the document…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Retrieving relevant clauses and generating answer…"):
        try:
            api_resp = requests.post(
                f"{API_URL}/chat",
                json={"query": prompt, "document_id": doc_id, "service_name": service_name},
                headers=_auth_headers(),
                timeout=120,
            )
            api_resp.raise_for_status()
            response = api_resp.json()
        except Exception as e:
            response = {"answer": f"API Error: {e}", "cited_sources": [], "all_retrieved": []}

    answer_text = response["answer"]
    cited       = response.get("cited_sources", [])
    all_ret     = response.get("all_retrieved", [])

    st.session_state.messages.append({
        "role":          "assistant",
        "content":       answer_text,
        "cited_sources": cited,
        "all_retrieved": all_ret,
    })

    with st.chat_message("assistant"):
        rendered = render_cited_text(answer_text, cited) if cited else answer_text
        st.markdown(rendered, unsafe_allow_html=True)
        if cited:
            show_source_expanders(cited, prefix="  ")
        if all_ret and len(all_ret) > len(cited):
            with st.expander(f"📚 All {len(all_ret)} retrieved chunks"):
                show_source_expanders(all_ret)

# ─────────────────────────────────────────────────────────────────────────────
# Reset
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔄 Start Over")
st.caption("Clear current document context and analyze a new one.")
if st.button("🗑️ Clear & Start Over", type="primary"):
    for key in ["current_doc_id", "current_source", "source_type", "messages", "summary_data"]:
        st.session_state.pop(key, None)
    for tmp in ["temp_tos.pdf", "temp_webpage.txt"]:
        if os.path.exists(tmp):
            os.remove(tmp)
    st.rerun()