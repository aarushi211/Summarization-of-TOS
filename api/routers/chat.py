import json
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from api.core import database
from api.core.security import get_current_user, limiter
from api.schemas.chat import ChatRequest
from src.RAG.schemas import SessionState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["AI Engine"])

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

def _get_state(user_id: str, doc_id: str) -> SessionState:
    namespace = f"{user_id}_{doc_id}".replace("-", "_")
    state = SessionState(pinecone_namespace=namespace, document_id=doc_id)
    if database.supa_admin:
        try:
            res = database.supa_admin.table("documents").select("service_name, doc_type").eq("id", doc_id).single().execute()
            if res.data:
                state.service_name = res.data.get("service_name", "Unknown")
                state.doc_type = res.data.get("doc_type", "Terms")
        except: pass
    return state

@router.get("/summary/{document_id}")
async def get_summary(document_id: str, user: dict = Depends(get_current_user)):
    if not database.shared_assistant:
        raise HTTPException(status_code=503, detail="AI engine not loaded")
    
    state = _get_state(user["user_id"], document_id)
    
    async def summary_gen():
        for msg in database.shared_assistant.generate_global_summary_stream(state):
            if msg["type"] == "topic_ready" and database.supa_admin:
                try:
                    data = msg["data"]
                    database.supa_admin.table("summaries").insert({
                        "document_id": document_id,
                        "user_id": user["user_id"],
                        "topic_label": data["label"],
                        "summary_text": data["summary"],
                        "sources": data["sources"]
                    }).execute()
                except Exception as e:
                    logger.error("Failed to save summary topic: %s", e)
            yield _sse(msg)

    return StreamingResponse(summary_gen(), media_type="text/event-stream")

@router.post("/query")
async def ask_question(req: ChatRequest, user: dict = Depends(get_current_user)):
    if not database.shared_assistant:
        raise HTTPException(status_code=503, detail="AI engine not loaded")
    
    state = _get_state(user["user_id"], req.document_id)
    
    async def chat_gen():
        full_response = ""
        sources = []
        
        # Save user message immediately
        if database.supa_admin:
            database.supa_admin.table("chats").insert({
                "document_id": req.document_id,
                "user_id": user["user_id"],
                "role": "user",
                "content": req.query
            }).execute()

        for msg in database.shared_assistant.answer_question_stream(req.query, state):
            if msg["type"] == "token":
                full_response += msg["data"]
            elif msg["type"] == "sources":
                sources = msg["data"]
            elif msg["type"] == "done" and database.supa_admin:
                # Save assistant response when complete
                try:
                    database.supa_admin.table("chats").insert({
                        "document_id": req.document_id,
                        "user_id": user["user_id"],
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources
                    }).execute()
                except Exception as e:
                    logger.error("Failed to save chat response: %s", e)
            yield _sse(msg)

    return StreamingResponse(chat_gen(), media_type="text/event-stream")
