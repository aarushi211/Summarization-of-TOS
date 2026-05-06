"""
api/routers/ingest.py
"""

import logging
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from api.core import database
from api.core.config import settings
from api.core.security import get_current_user
from api.utils.validation import detect_mime, assert_safe_url
from src.RAG.processors import sanitise_label
from src.RAG.schemas import SessionState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])


# ── Upload validation ─────────────────────────────────────────────────────────

async def _validate_upload(file: UploadFile) -> bytes:
    content = await file.read()
    if len(content) > settings.MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    mime = detect_mime(content)
    if mime not in settings.ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported type {mime}")
    return content


# ── Background ingest ─────────────────────────────────────────────────────────

def _save_file_locally(temp_path: str, user_id: str, doc_id: str) -> str:
    """
    In desktop mode, copy the uploaded file into ~/.tos-summarizer/uploads/
    instead of uploading to S3. Returns the final local path.
    """
    dest_dir = settings.DESKTOP_DATA_DIR / "uploads" / user_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{doc_id}.pdf"
    shutil.copy2(temp_path, dest)
    return str(dest)


def _background_ingest(
    temp_path: str,
    user_id: str,
    doc_id: str,
    namespace: str,
    service_name: str,
    doc_type: str,
):
    """
    Ingest a document into the vector store in the background.

    Changes vs original:
    1. Structured logging — every log includes user_id and doc_id so you can
       find all logs for a specific failed document in Cloud Logging instantly.
    2. Retry with exponential backoff — Pinecone timeouts are transient.
       We retry up to 3 times (1s → 2s → 4s) before marking the doc as error.
    3. Desktop vs server branching — S3 becomes local filesystem, Pinecone
       becomes ChromaDB, but the router code above is unchanged.
    4. Granular error states — DB is updated with an error_reason so the
       frontend can show "why did this fail" instead of just a red dot.
    """
    log_ctx = {"user_id": user_id, "document_id": doc_id}
    logger.info("background ingest started", extra=log_ctx)

    MAX_RETRIES = 3

    try:
        state = SessionState(
            pinecone_namespace=namespace,
            document_id=doc_id,
            service_name=service_name,
            doc_type=doc_type,
        )

        # ── Step 1: File storage ──────────────────────────────────────────────
        # Desktop: copy to local uploads folder (no S3 account needed)
        # Server:  upload to S3 (archival — not critical path, don't fail ingest)
        if settings.is_desktop:
            local_path = _save_file_locally(temp_path, user_id, doc_id)
            logger.info("file saved locally", extra={**log_ctx, "path": local_path})
        else:
            if database.s3_client:
                s3_key = f"uploads/{user_id}/{doc_id}.pdf"
                try:
                    database.s3_client.upload_file(
                        temp_path, settings.AWS_S3_BUCKET_NAME, s3_key
                    )
                    logger.info("uploaded to S3", extra={**log_ctx, "s3_key": s3_key})
                except Exception as e:
                    # S3 is archival — log and continue. Ingest can succeed without it.
                    logger.warning(
                        "S3 upload failed (non-fatal)",
                        extra={**log_ctx, "error": str(e)},
                    )

        # ── Step 2: Vector ingestion (critical path) ──────────────────────────
        # This is the operation most likely to fail transiently (Pinecone timeout,
        # ChromaDB lock contention). We retry with exponential backoff.
        if not database.shared_assistant:
            raise RuntimeError("AI assistant not initialised — cannot ingest document.")

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                database.shared_assistant.ingest_document(temp_path, state)
                logger.info(
                    "vector ingestion complete",
                    extra={**log_ctx, "attempt": attempt + 1},
                )
                last_error = None
                break  # Success — exit retry loop
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        "vector ingestion failed, retrying",
                        extra={
                            **log_ctx,
                            "attempt": attempt + 1,
                            "retry_in_s": wait,
                            "error": str(e),
                        },
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "vector ingestion failed after all retries",
                        extra={**log_ctx, "attempts": MAX_RETRIES, "error": str(e)},
                        exc_info=True,
                    )

        if last_error:
            # All retries exhausted — mark document as failed with a reason
            # so the frontend can surface a useful message to the user
            if database.supa_admin:
                database.supa_admin.table("documents").update({
                    "status": "error",
                    "error_reason": f"Ingestion failed: {str(last_error)[:200]}",
                }).eq("id", doc_id).execute()
            return

        # ── Step 3: Mark document as ready ────────────────────────────────────
        if database.supa_admin:
            database.supa_admin.table("documents").update(
                {"status": "ready"}
            ).eq("id", doc_id).execute()

        logger.info("document ready", extra=log_ctx)

    except Exception as e:
        # Catch-all for unexpected errors (programming bugs, not transient failures)
        logger.error(
            "unexpected ingest error",
            extra={**log_ctx, "error": str(e)},
            exc_info=True,
        )
        if database.supa_admin:
            database.supa_admin.table("documents").update({
                "status": "error",
                "error_reason": f"Unexpected error: {str(e)[:200]}",
            }).eq("id", doc_id).execute()

    finally:
        # Always clean up the temp file
        if Path(temp_path).exists():
            Path(temp_path).unlink(missing_ok=True)


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    service_name: str = Form("Unknown"),
    doc_type: str = Form("Terms of Service"),
    user: dict = Depends(get_current_user),
):
    content = await _validate_upload(file)
    doc_id = str(uuid.uuid4())
    namespace = f"{user['user_id']}_{doc_id}".replace("-", "_")

    # In server mode, record the S3 key for later reference.
    # In desktop mode, the key is the local path (written in background task).
    s3_key = (
        f"local:{user['user_id']}/{doc_id}.pdf"
        if settings.is_desktop
        else f"uploads/{user['user_id']}/{doc_id}.pdf"
    )

    if database.supa_admin:
        database.supa_admin.table("documents").insert({
            "id": doc_id,
            "user_id": user["user_id"],
            "filename": file.filename,
            "service_name": sanitise_label(service_name),
            "doc_type": sanitise_label(doc_type),
            "s3_key": s3_key,
            "pinecone_ns": namespace,
            "status": "processing",
        }).execute()

    # Write to temp file and kick off background task
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    background_tasks.add_task(
        _background_ingest,
        tmp_path,
        user["user_id"],
        doc_id,
        namespace,
        service_name,
        doc_type,
    )

    return {"document_id": doc_id, "status": "processing"}