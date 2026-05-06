"""
api/routers/chat.py
"""

import json
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from postgrest.exceptions import APIError as SupabaseAPIError

from api.core import database
from api.core.config import settings
from api.core.security import get_current_user
from api.schemas.chat import ChatRequest
from src.RAG.schemas import SessionState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["AI Engine"])


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _get_state(request: Request, user_id: str, doc_id: str) -> SessionState:
    """
    Build a SessionState for this request.
    In desktop mode, user_id is always "local-user" — no auth needed.
    """
    rid = getattr(request.state, "request_id", "-")
    namespace = f"{user_id}_{doc_id}".replace("-", "_")
    state = SessionState(pinecone_namespace=namespace, document_id=doc_id)

    if not database.supa_admin:
        return state

    try:
        res = (
            database.supa_admin
            .table("documents")
            .select("service_name, doc_type")
            .eq("id", doc_id)
            .single()
            .execute()
        )
        if res.data:
            state.service_name = res.data.get("service_name", "Unknown")
            state.doc_type = res.data.get("doc_type", "Terms")
    except (SupabaseAPIError, Exception) as e:
        # SupabaseAPIError = row not found (expected in desktop SQLite too)
        # Any other exception = log with context but don't crash the request
        is_not_found = isinstance(e, SupabaseAPIError)
        if not is_not_found:
            logger.warning(
                "could not fetch document metadata",
                extra={"request_id": rid, "document_id": doc_id},
                exc_info=True,
            )

    return state


def _try_save(rid: str, doc_id: str, row: dict, context: str):
    """
    Helper to persist a DB row with structured error logging.
    Shared by both summary and chat generators to avoid duplication.
    """
    try:
        table = "summaries" if "topic_label" in row else "chats"
        database.supa_admin.table(table).insert(row).execute()
    except Exception:
        logger.error(
            "failed to persist %s", context,
            extra={"request_id": rid, "document_id": doc_id},
            exc_info=True,
        )


# ── Summary stream ────────────────────────────────────────────────────────────

@router.get("/summary/{document_id}")
async def get_summary(
    document_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    if not database.shared_assistant:
        raise HTTPException(status_code=503, detail="AI engine not loaded")

    rid = getattr(request.state, "request_id", "-")
    state = _get_state(request, user["user_id"], document_id)

    async def summary_gen():
        # ── Streaming error boundary ──────────────────────────────────────────
        # If the generator raises mid-stream (model crash, OOM, etc.) we catch
        # it here and yield a structured error + done event so the frontend
        # always receives a terminal message and never hangs waiting.
        try:
            for msg in database.shared_assistant.generate_global_summary_stream(state):
                if msg["type"] == "topic_ready" and database.supa_admin:
                    data = msg["data"]
                    _try_save(rid, document_id, {
                        "document_id": document_id,
                        "user_id": user["user_id"],
                        "topic_label": data["label"],
                        "summary_text": data["summary"],
                        "sources": data["sources"],
                    }, context=f"summary topic '{data['label']}'")
                yield _sse(msg)

        except Exception:
            logger.error(
                "summary stream failed",
                extra={"request_id": rid, "document_id": document_id},
                exc_info=True,
            )
            # Always yield error + done so the frontend terminates cleanly.
            # Without this the client waits until its own timeout fires.
            yield _sse({"type": "error", "data": "Summary generation failed. Please try again."})
            yield _sse({"type": "done"})

    return StreamingResponse(summary_gen(), media_type="text/event-stream")


# ── Chat stream ───────────────────────────────────────────────────────────────

@router.post("/query")
async def ask_question(
    req: ChatRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    if not database.shared_assistant:
        raise HTTPException(status_code=503, detail="AI engine not loaded")

    rid = getattr(request.state, "request_id", "-")
    state = _get_state(request, user["user_id"], req.document_id)

    async def chat_gen():
        full_response = ""
        sources = []

        # Persist user message before streaming starts
        if database.supa_admin:
            _try_save(rid, req.document_id, {
                "document_id": req.document_id,
                "user_id": user["user_id"],
                "role": "user",
                "content": req.query,
            }, context="user message")

        # ── Streaming error boundary ──────────────────────────────────────────
        # Wrapping the entire generator in try/except means:
        # - A Pinecone timeout mid-retrieval → clean error to frontend
        # - An OOM during inference → clean error to frontend
        # - A programming error → clean error to frontend + full traceback in logs
        #
        # Without this, the SSE connection just closes mid-stream. The frontend
        # (useSseChat.ts) catches the network error but can't show a meaningful
        # message to the user. With this, it receives {"type":"error","data":"..."}
        # which the hook already handles correctly (see useSseChat.ts error case).
        try:
            for msg in database.shared_assistant.answer_question_stream(req.query, state):
                if msg["type"] == "token":
                    full_response += msg["data"]
                elif msg["type"] == "sources":
                    sources = msg["data"]
                elif msg["type"] == "done" and database.supa_admin:
                    _try_save(rid, req.document_id, {
                        "document_id": req.document_id,
                        "user_id": user["user_id"],
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources,
                    }, context="assistant response")
                yield _sse(msg)

        except Exception:
            logger.error(
                "chat stream failed",
                extra={
                    "request_id": rid,
                    "document_id": req.document_id,
                    "partial_response_len": len(full_response),
                },
                exc_info=True,
            )
            yield _sse({"type": "error", "data": "AI generation failed. Please try again."})
            yield _sse({"type": "done"})

    return StreamingResponse(chat_gen(), media_type="text/event-stream")