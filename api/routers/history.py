import logging
from fastapi import APIRouter, Depends, HTTPException
from api.core import database
from api.core.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/history", tags=["History"])

@router.get("/documents")
async def get_history(user: dict = Depends(get_current_user)):
    if not database.supa_admin:
        return []
    try:
        res = (
            database.supa_admin.table("documents")
            .select("*")
            .eq("user_id", user["user_id"])
            .order("created_at", desc=True)
            .execute()
        )
        return {"documents": res.data}
    except Exception as exc:
        logger.error("Failed to fetch history: %s", exc)
        raise HTTPException(status_code=500, detail="Database error")

@router.get("/documents/{document_id}/status")
async def get_document_status(document_id: str, user: dict = Depends(get_current_user)):
    if not database.supa_admin:
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        res = database.supa_admin.table("documents").select("status").eq("id", document_id).eq("user_id", user["user_id"]).single().execute()
        return res.data
    except Exception as exc:
        logger.error("Status check failed: %s", exc)
        return {"status": "error", "error_reason": str(exc)}

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str, user: dict = Depends(get_current_user)):
    if not database.supa_admin:
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        # Delete from DB (S3/Pinecone cleanup can be async or via separate job)
        database.supa_admin.table("documents").delete().eq("id", document_id).eq("user_id", user["user_id"]).execute()
        return {"status": "deleted"}
    except Exception as exc:
        logger.error("Deletion failed: %s", exc)
        raise HTTPException(status_code=500, detail="Deletion failed")

@router.get("/chats/{document_id}")
async def get_chat_history(document_id: str, user: dict = Depends(get_current_user)):
    if not database.supa_admin:
        return {"sessions": []}
    try:
        res = (
            database.supa_admin.table("chats")
            .select("*")
            .eq("document_id", document_id)
            .eq("user_id", user["user_id"])
            .order("created_at", desc=True)
            .execute()
        )
        # Group into a single session for frontend compatibility
        return {"sessions": [{"messages": res.data[::-1]}]} if res.data else {"sessions": []}
    except Exception as exc:
        logger.error("Failed to load chat history: %s", exc)
        return {"sessions": []}

@router.get("/summaries/{document_id}")
async def get_summary_history(document_id: str, user: dict = Depends(get_current_user)):
    if not database.supa_admin:
        return []
    try:
        res = (
            database.supa_admin.table("summaries")
            .select("*")
            .eq("document_id", document_id)
            .eq("user_id", user["user_id"])
            .order("created_at", desc=True)
            .execute()
        )
        # Convert to the format expected by the frontend TopicResult
        return [
            {
                "label": r["topic_label"],
                "summary": r["summary_text"],
                "sources": r["sources"]
            }
            for r in res.data
        ]
    except Exception as exc:
        logger.error("Failed to load summaries: %s", exc)
        return []
