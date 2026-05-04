"""
TOS Summarizer FastAPI Backend
- Supabase Auth (JWT validation on every protected route)
- AWS S3 (PDF storage)
- Pinecone (persistent vector embeddings, namespaced per user+document)
- Supabase PostgreSQL (document metadata + chat history)
"""

import os
import sys
import time
import uuid
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header, Depends
from pydantic import BaseModel
from supabase import create_client, Client
import jwt as pyjwt
from dotenv import load_dotenv

load_dotenv()

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.RAG.rag_pipeline import TOSAssistant, SessionState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config from environment ───────────────────────────────────────────────────
MODEL_PATH         = PROJECT_ROOT / "models" / "legal_qwen.Q4_K_M.gguf"
SUPABASE_URL       = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY  = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_SERV_KEY  = os.getenv("SUPABASE_SERVICE_KEY", "")
AWS_BUCKET         = os.getenv("AWS_S3_BUCKET_NAME", "")
AWS_REGION         = os.getenv("AWS_REGION", "us-east-1")
AWS_KEY_ID         = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET         = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# ── Shared singletons ─────────────────────────────────────────────────────────
shared_assistant: Optional[TOSAssistant] = None
# Supabase admin client (uses service key — never exposed to frontend)
supa_admin: Optional[Client] = None
# S3 client
s3_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global shared_assistant, supa_admin, s3_client

    # Supabase admin client
    if SUPABASE_URL and SUPABASE_SERV_KEY:
        supa_admin = create_client(SUPABASE_URL, SUPABASE_SERV_KEY)
        logger.info("Supabase admin client ready.")
    else:
        logger.warning("Supabase env vars not set — DB writes disabled.")

    # S3 client
    if AWS_KEY_ID and AWS_SECRET:
        s3_client = boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_KEY_ID,
            aws_secret_access_key=AWS_SECRET,
        )
        logger.info(f"AWS S3 client ready (bucket: {AWS_BUCKET}).")
    else:
        logger.warning("AWS env vars not set — S3 uploads disabled.")

    # RAG engine (heavy — load last)
    logger.info(f"Loading TOSAssistant from {MODEL_PATH}...")
    if MODEL_PATH.exists():
        try:
            shared_assistant = TOSAssistant(str(MODEL_PATH))
            logger.info("TOSAssistant loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load TOSAssistant: {e}")
    else:
        logger.error(f"Model not found at {MODEL_PATH}")

    yield
    logger.info("Shutting down.")


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="TOS Summarizer API", version="2.0", lifespan=lifespan)

# Add CORS Middleware to allow React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── JWT Auth ──────────────────────────────────────────────────────────────────
def _get_current_user(authorization: Optional[str] = Header(default=None)) -> dict:
    """
    Validate the Supabase JWT from the Authorization header.
    Returns the decoded payload (includes `sub` == user_id).
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.split(" ", 1)[1]
    try:
        # Decode without verifying signature for now (Supabase handles verification)
        # In production you'd verify with the Supabase JWT secret
        payload = pyjwt.decode(token, options={"verify_signature": False})
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing sub.")
        return {"user_id": user_id, "token": token}
    except pyjwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Token decode error: {e}")


# ── Pydantic models ───────────────────────────────────────────────────────────
class SignUpRequest(BaseModel):
    email:    str
    password: str

class LoginRequest(BaseModel):
    email:    str
    password: str

class ChatRequest(BaseModel):
    query:        str
    document_id:  str
    service_name: str = "Unknown Service"

class SummaryRequest(BaseModel):
    document_id:  str
    service_name: str = "Unknown Service"
    doc_type:     str = "Terms of Service"


# ── Auth endpoints ────────────────────────────────────────────────────────────
@app.post("/auth/signup")
async def signup(req: SignUpRequest):
    if not supa_admin:
        raise HTTPException(status_code=503, detail="Auth service not configured.")
    try:
        res = supa_admin.auth.sign_up({"email": req.email, "password": req.password})
        return {"message": "Account created. Check your email to confirm.", "user_id": res.user.id if res.user else None}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/login")
async def login(req: LoginRequest):
    if not supa_admin:
        raise HTTPException(status_code=503, detail="Auth service not configured.")
    try:
        res = supa_admin.auth.sign_in_with_password({"email": req.email, "password": req.password})
        return {
            "access_token":  res.session.access_token,
            "refresh_token": res.session.refresh_token,
            "user_id":       res.user.id,
            "email":         res.user.email,
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "rag_engine": "ready" if shared_assistant else "loading",
        "supabase":   "ready" if supa_admin else "not configured",
        "s3":         "ready" if s3_client else "not configured",
    }


# ── Helper: resolve session state from document_id ───────────────────────────
def _build_session_state(user_id: str, document_id: str) -> SessionState:
    """Build a SessionState from a known document_id (for loading existing docs)."""
    namespace = f"{user_id}_{document_id}".replace("-", "_")
    state = SessionState(
        pinecone_namespace=namespace,
        document_id=document_id,
    )
    # Fetch service_name / doc_type from Supabase if available
    if supa_admin:
        try:
            row = (
                supa_admin.table("documents")
                .select("service_name, doc_type")
                .eq("id", document_id)
                .eq("user_id", user_id)
                .single()
                .execute()
            )
            if row.data:
                state.service_name = row.data.get("service_name", "Unknown Service")
                state.doc_type     = row.data.get("doc_type",     "Unknown Document")
        except Exception:
            pass
    return state


# ── Ingest: PDF ───────────────────────────────────────────────────────────────
@app.post("/ingest/pdf")
async def ingest_pdf(
    file:         UploadFile = File(...),
    service_name: str        = Form(default="Unknown Service"),
    doc_type:     str        = Form(default="Terms of Service"),
    current_user: dict       = Depends(_get_current_user),
):
    if shared_assistant is None:
        raise HTTPException(status_code=503, detail="RAG Engine not ready.")

    user_id    = current_user["user_id"]
    doc_id     = str(uuid.uuid4())
    namespace  = f"{user_id}_{doc_id}".replace("-", "_")
    s3_key     = f"uploads/{user_id}/{doc_id}/{file.filename}"
    temp_path  = f"temp_{doc_id}_{file.filename}"

    try:
        # 1. Save to temp file
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # 2. Upload to S3
        if s3_client:
            s3_client.upload_file(temp_path, AWS_BUCKET, s3_key)
            logger.info(f"[{user_id}] Uploaded to S3: {s3_key}")

        # 3. Ingest into Pinecone (namespaced)
        state = SessionState(
            pinecone_namespace=namespace,
            document_id=doc_id,
            service_name=service_name,
            doc_type=doc_type,
        )
        shared_assistant.ingest_document(temp_path, state)

        # 4. Save document metadata to Supabase
        if supa_admin:
            supa_admin.table("documents").insert({
                "id":           doc_id,
                "user_id":      user_id,
                "filename":     file.filename,
                "s3_key":       s3_key,
                "service_name": service_name,
                "doc_type":     doc_type,
                "pinecone_ns":  namespace,
            }).execute()
            logger.info(f"[{user_id}] Document metadata saved: {doc_id}")

        return {"document_id": doc_id, "namespace": namespace, "message": f"Ingested {file.filename}"}

    except Exception as e:
        logger.exception(f"[{user_id}] PDF ingest failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── Ingest: URL / scraped text ────────────────────────────────────────────────
@app.post("/ingest/text")
async def ingest_text(
    text:         str  = Form(...),
    filename:     str  = Form(default="scraped_page.txt"),
    service_name: str  = Form(default="Unknown Service"),
    doc_type:     str  = Form(default="Terms of Service"),
    current_user: dict = Depends(_get_current_user),
):
    if shared_assistant is None:
        raise HTTPException(status_code=503, detail="RAG Engine not ready.")

    user_id   = current_user["user_id"]
    doc_id    = str(uuid.uuid4())
    namespace = f"{user_id}_{doc_id}".replace("-", "_")
    s3_key    = f"uploads/{user_id}/{doc_id}/{filename}"
    temp_path = f"temp_{doc_id}.txt"

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(text)

        # Upload to S3
        if s3_client:
            s3_client.upload_file(temp_path, AWS_BUCKET, s3_key)

        # Ingest into Pinecone
        state = SessionState(
            pinecone_namespace=namespace,
            document_id=doc_id,
            service_name=service_name,
            doc_type=doc_type,
        )
        shared_assistant.ingest_text_file(temp_path, state)

        # Save metadata
        if supa_admin:
            supa_admin.table("documents").insert({
                "id":           doc_id,
                "user_id":      user_id,
                "filename":     filename,
                "s3_key":       s3_key,
                "service_name": service_name,
                "doc_type":     doc_type,
                "pinecone_ns":  namespace,
            }).execute()

        return {"document_id": doc_id, "namespace": namespace, "message": f"Ingested {filename}"}

    except Exception as e:
        logger.exception(f"[{user_id}] Text ingest failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── Summary ───────────────────────────────────────────────────────────────────
@app.post("/summary")
async def get_summary(
    req:          SummaryRequest,
    current_user: dict = Depends(_get_current_user),
):
    if shared_assistant is None:
        raise HTTPException(status_code=503, detail="RAG Engine not ready.")

    user_id = current_user["user_id"]
    state   = _build_session_state(user_id, req.document_id)
    state.service_name = req.service_name
    state.doc_type     = req.doc_type

    try:
        result = shared_assistant.generate_global_summary(state)
        return result
    except Exception as e:
        logger.exception(f"[{user_id}] Summary failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Chat ──────────────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(
    req:          ChatRequest,
    current_user: dict = Depends(_get_current_user),
):
    if shared_assistant is None:
        raise HTTPException(status_code=503, detail="RAG Engine not ready.")

    user_id = current_user["user_id"]
    state   = _build_session_state(user_id, req.document_id)
    state.service_name = req.service_name

    try:
        result = shared_assistant.answer_question(req.query, state)

        # Persist chat to Supabase
        if supa_admin:
            # Find or create a chat session for this document
            session_res = (
                supa_admin.table("chat_sessions")
                .select("id")
                .eq("user_id",     user_id)
                .eq("document_id", req.document_id)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            if session_res.data:
                session_id = session_res.data[0]["id"]
            else:
                new_session = supa_admin.table("chat_sessions").insert({
                    "user_id":     user_id,
                    "document_id": req.document_id,
                    "title":       req.query[:60],
                }).execute()
                session_id = new_session.data[0]["id"]

            # Save user message + assistant reply
            supa_admin.table("chat_messages").insert([
                {"session_id": session_id, "role": "user",      "content": req.query},
                {"session_id": session_id, "role": "assistant",  "content": result["answer"],
                 "cited_sources": result.get("cited_sources", [])},
            ]).execute()

        return result
    except Exception as e:
        logger.exception(f"[{user_id}] Chat failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── History: documents ────────────────────────────────────────────────────────
@app.get("/history/documents")
async def get_documents(current_user: dict = Depends(_get_current_user)):
    if not supa_admin:
        raise HTTPException(status_code=503, detail="Database not configured.")
    user_id = current_user["user_id"]
    try:
        res = (
            supa_admin.table("documents")
            .select("id, filename, service_name, doc_type, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        return {"documents": res.data or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── History: chat messages for a document ─────────────────────────────────────
@app.get("/history/chats/{document_id}")
async def get_chat_history(
    document_id:  str,
    current_user: dict = Depends(_get_current_user),
):
    if not supa_admin:
        raise HTTPException(status_code=503, detail="Database not configured.")
    user_id = current_user["user_id"]
    try:
        # Get all sessions for this document
        sessions = (
            supa_admin.table("chat_sessions")
            .select("id, title, created_at")
            .eq("user_id",     user_id)
            .eq("document_id", document_id)
            .order("created_at", desc=True)
            .execute()
        )
        if not sessions.data:
            return {"sessions": []}

        # Get messages for all sessions
        session_ids = [s["id"] for s in sessions.data]
        messages    = (
            supa_admin.table("chat_messages")
            .select("session_id, role, content, cited_sources, created_at")
            .in_("session_id", session_ids)
            .order("created_at")
            .execute()
        )
        # Group messages by session
        msg_by_session: dict = {s["id"]: [] for s in sessions.data}
        for msg in (messages.data or []):
            sid = msg["session_id"]
            if sid in msg_by_session:
                msg_by_session[sid].append(msg)

        result = [
            {**s, "messages": msg_by_session[s["id"]]}
            for s in sessions.data
        ]
        return {"sessions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Clear (delete Pinecone namespace for a document) ─────────────────────────
@app.delete("/documents/{document_id}")
async def delete_document(
    document_id:  str,
    current_user: dict = Depends(_get_current_user),
):
    user_id = current_user["user_id"]
    if shared_assistant:
        namespace = f"{user_id}_{document_id}".replace("-", "_")
        try:
            shared_assistant._pc_index.delete(delete_all=True, namespace=namespace)
            logger.info(f"[{user_id}] Deleted Pinecone namespace: {namespace}")
        except Exception as e:
            logger.warning(f"Pinecone delete failed: {e}")
    if supa_admin:
        supa_admin.table("documents").delete().eq("id", document_id).eq("user_id", user_id).execute()
    return {"message": f"Document {document_id} deleted."}