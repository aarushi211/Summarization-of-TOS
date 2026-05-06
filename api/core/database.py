"""
api/core/database.py
"""

import sys
import logging
from pathlib import Path
from typing import Optional

from api.core.config import settings

logger = logging.getLogger(__name__)

supa_admin = None
s3_client = None
shared_assistant = None


def init_db():
    global supa_admin

    if settings.is_desktop:
        from api.core.local_db import LocalDB
        db_path = settings.DESKTOP_DATA_DIR / "data.db"
        supa_admin = LocalDB(db_path)
        logger.info("Desktop mode: LocalDB (SQLite) initialised at %s", db_path)
    else:
        if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_KEY:
            from supabase import create_client
            supa_admin = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
            logger.info("Server mode: Supabase client ready")
        else:
            logger.warning("Supabase env vars not set — DB features disabled")


def init_storage():
    global s3_client

    if settings.is_desktop:
        upload_dir = settings.DESKTOP_DATA_DIR / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Desktop mode: local file storage at %s", upload_dir)
    else:
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            import boto3
            s3_client = boto3.client(
                "s3",
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            )
            logger.info("Server mode: S3 client ready")
        else:
            logger.warning("AWS env vars not set — S3 storage disabled")


def init_assistant():
    global shared_assistant

    logger.info("Loading TOSAssistant from %s...", settings.MODEL_PATH)

    if not settings.MODEL_PATH.exists():
        msg = "Model file not found at %s."
        if settings.IS_PRODUCTION:
            logger.critical(
                msg + " Exiting — cannot serve traffic without model.",
                settings.MODEL_PATH,
            )
            sys.exit(1)
        else:
            logger.warning(msg + " AI features disabled.", settings.MODEL_PATH)
            return

    try:
        from src.RAG.engine import TOSAssistant

        if settings.is_desktop:
            shared_assistant = TOSAssistant(
                model_path=str(settings.MODEL_PATH),
                pinecone_api_key="",
                index_name=settings.CHROMA_COLLECTION_NAME,
                data_dir=str(settings.DESKTOP_DATA_DIR / "chroma"),
                use_local_vectorstore=True,
                n_gpu_layers=settings.N_GPU_LAYERS,
            )
            logger.info("Desktop mode: TOSAssistant loaded with ChromaDB")
        else:
            shared_assistant = TOSAssistant(
                model_path=str(settings.MODEL_PATH),
                pinecone_api_key=settings.PINECONE_API_KEY,
                index_name=settings.PINECONE_INDEX_NAME,
                dimension=settings.PINECONE_DIMENSION,
                cloud=settings.PINECONE_CLOUD,
                region=settings.PINECONE_REGION,
                n_gpu_layers=settings.N_GPU_LAYERS,
            )
            logger.info("Server mode: TOSAssistant loaded with Pinecone")

    except Exception as e:
        logger.critical("Failed to load TOSAssistant: %s", e, exc_info=True)
        if settings.IS_PRODUCTION:
            sys.exit(1)
        else:
            logger.warning("Continuing without AI model (dev/desktop mode).")