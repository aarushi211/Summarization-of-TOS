import json
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from postgrest.exceptions import APIError as SupabaseAPIError

from api.core import database
from api.core.security import get_current_user, limiter
from api.schemas.chat import ChatRequest
from src.RAG.schemas import SessionState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["AI Engine"])


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _get_state(request: Request, user_id: str, doc_id: str) -> SessionState:
    """
    Build a SessionState for this request.

    ── FIX: Replace bare `except: pass` with specific exception handling. ────
    The original code silently swallowed every exception here, including
    programming errors, auth failures, and network timeouts. The bare except
    made it impossible to distinguish "document metadata not found" (expected)
    from "Supabase is down" (needs alerting).

    Now we catch only the exceptions we expect:
    - SupabaseAPIError: the row doesn't exist or RLS blocked the query. Fine —
      we proceed with default state values.
    - Exception: anything else (network timeout, auth error) is logged as a
      warning with the request_id so it shows up in Cloud Logging, but we
      still continue rather than crashing the request.
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
    except SupabaseAPIError:
        # Row not found or RLS denied — not an error, just use defaults
        pass
    except Exception:
        # Unexpected error (network, auth) — log it but don't crash the request
        logger.warning(
            "could not fetch document metadata",
            extra={"request_id": rid, "document_id": doc_id},
            exc_info=True,
        )

    return state


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
        for msg in database.shared_assistant.generate_global_summary_stream(state):
            if msg["type"] == "topic_ready" and database.supa_admin:
                data = msg["data"]
                try:
                    database.supa_admin.table("summaries").insert({
                        "document_id": document_id,
                        "user_id": user["user_id"],
                        "topic_label": data["label"],
                        "summary_text": data["summary"],
                        "sources": data["sources"],
                    }).execute()
                except SupabaseAPIError as e:
                    # ── FIX: Log Supabase write failures with request_id. ──
                    # Previously: logger.error("Failed to save summary: %s", e)
                    # That log had no request_id, no topic context, and no
                    # indication of which user was affected. This version gives
                    # you everything you need to reproduce the failure.
                    logger.error(
                        "failed to persist summary topic",
                        extra={
                            "request_id": rid,
                            "document_id": document_id,
                            "topic": data.get("label"),
                            "error": str(e),
                        },
                    )
                except Exception:
                    logger.error(
                        "unexpected error persisting summary topic",
                        extra={"request_id": rid, "document_id": document_id},
                        exc_info=True,
                    )
            yield _sse(msg)

    return StreamingResponse(summary_gen(), media_type="text/event-stream")


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

        if database.supa_admin:
            try:
                database.supa_admin.table("chats").insert({
                    "document_id": req.document_id,
                    "user_id": user["user_id"],
                    "role": "user",
                    "content": req.query,
                }).execute()
            except Exception:
                logger.warning(
                    "failed to persist user message",
                    extra={"request_id": rid, "document_id": req.document_id},
                    exc_info=True,
                )

        for msg in database.shared_assistant.answer_question_stream(req.query, state):
            if msg["type"] == "token":
                full_response += msg["data"]
            elif msg["type"] == "sources":
                sources = msg["data"]
            elif msg["type"] == "done" and database.supa_admin:
                try:
                    database.supa_admin.table("chats").insert({
                        "document_id": req.document_id,
                        "user_id": user["user_id"],
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources,
                    }).execute()
                except SupabaseAPIError as e:
                    logger.error(
                        "failed to persist assistant response",
                        extra={
                            "request_id": rid,
                            "document_id": req.document_id,
                            "response_length": len(full_response),
                            "error": str(e),
                        },
                    )
                except Exception:
                    logger.error(
                        "unexpected error persisting assistant response",
                        extra={"request_id": rid, "document_id": req.document_id},
                        exc_info=True,
                    )
            yield _sse(msg)

    return StreamingResponse(chat_gen(), media_type="text/event-stream")