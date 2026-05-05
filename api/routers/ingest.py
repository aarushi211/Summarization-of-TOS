import logging
import uuid
import tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from api.core import database
from api.core.security import get_current_user, limiter
from api.core.config import settings
from api.utils.validation import detect_mime, assert_safe_url
from src.RAG.schemas import SessionState
from src.RAG.processors import sanitise_label

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])

async def _validate_upload(file: UploadFile) -> bytes:
    content = await file.read()
    if len(content) > settings.MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    mime = detect_mime(content)
    if mime not in settings.ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported type {mime}")
    return content

def _background_ingest(temp_path: str, user_id: str, doc_id: str, namespace: str, service_name: str, doc_type: str):
    try:
        state = SessionState(
            pinecone_namespace=namespace, 
            document_id=doc_id,
            service_name=service_name,
            doc_type=doc_type
        )
        if database.s3_client:
            try:
                s3_key = f"uploads/{user_id}/{doc_id}.pdf"
                database.s3_client.upload_file(temp_path, settings.AWS_S3_BUCKET_NAME, s3_key)
                logger.info("Uploaded %s to S3.", s3_key)
            except Exception as e:
                logger.error("S3 upload failed: %s", e)

        if database.shared_assistant:
            database.shared_assistant.ingest_document(temp_path, state)
            if database.supa_admin:
                database.supa_admin.table("documents").update({"status": "ready"}).eq("id", doc_id).execute()
    except Exception as e:
        logger.error("Background ingest failed: %s", e)
        if database.supa_admin:
            database.supa_admin.table("documents").update({"status": "error"}).eq("id", doc_id).execute()
    finally:
        if Path(temp_path).exists():
            Path(temp_path).unlink()

@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    service_name: str = Form("Unknown"),
    doc_type: str = Form("Terms of Service"),
    user: dict = Depends(get_current_user)
):
    content = await _validate_upload(file)
    doc_id = str(uuid.uuid4())
    
    s3_key = f"uploads/{user['user_id']}/{doc_id}.pdf"
    namespace = f"{user['user_id']}_{doc_id}".replace("-", "_")
    
    # Save to DB first
    if database.supa_admin:
        database.supa_admin.table("documents").insert({
            "id": doc_id,
            "user_id": user["user_id"],
            "filename": file.filename,
            "service_name": sanitise_label(service_name),
            "doc_type": sanitise_label(doc_type),
            "s3_key": s3_key,
            "pinecone_ns": namespace,
            "status": "processing"
        }).execute()

    # Save to temp and start background task
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    background_tasks.add_task(
        _background_ingest, tmp_path, user["user_id"], doc_id, namespace, service_name, doc_type
    )
    
    return {"document_id": doc_id, "status": "processing"}
