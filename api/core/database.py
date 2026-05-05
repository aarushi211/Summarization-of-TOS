import sys
import boto3
import logging
from typing import Optional
from supabase import create_client, Client
from api.core.config import settings
from src.RAG.engine import TOSAssistant

logger = logging.getLogger(__name__)

# Singletons
supa_admin: Optional[Client] = None
s3_client = None
shared_assistant: Optional[TOSAssistant] = None


def init_db():
    global supa_admin
    if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_KEY:
        supa_admin = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
        logger.info("Supabase admin client ready.")
    else:
        logger.warning("Supabase env vars not set.")


def init_s3():
    global s3_client
    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        s3_client = boto3.client(
            "s3", region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        logger.info("S3 client ready.")
    else:
        logger.warning("AWS env vars not set.")


def init_assistant():
    """
    Load the TOSAssistant singleton.

    ── FIX: Fail fast on model load failure. ────────────────────────────────
    Previously, a failed model load was logged and swallowed. Cloud Run health
    checks still passed, traffic was routed, and then 100% of inference
    requests failed silently at runtime.

    Now we call sys.exit(1) in production so Cloud Run sees a failed startup,
    retries the instance, and never routes traffic to a broken container.
    In non-production (local dev) we log a warning and continue so you can
    still reach /docs and /health while iterating without a model file.
    """
    global shared_assistant
    logger.info("Loading TOSAssistant from %s...", settings.MODEL_PATH)

    if not settings.MODEL_PATH.exists():
        msg = "Model file not found at %s."
        if settings.IS_PRODUCTION:
            logger.critical(msg + " Exiting — cannot serve traffic without model.", settings.MODEL_PATH)
            sys.exit(1)
        else:
            logger.warning(msg + " AI features disabled in dev mode.", settings.MODEL_PATH)
            return

    try:
        shared_assistant = TOSAssistant(
            model_path=str(settings.MODEL_PATH),
            pinecone_api_key=settings.PINECONE_API_KEY,
            index_name=settings.PINECONE_INDEX_NAME,
            dimension=settings.PINECONE_DIMENSION,
            cloud=settings.PINECONE_CLOUD,
            region=settings.PINECONE_REGION,
        )
        logger.info("TOSAssistant loaded successfully.")
    except Exception as e:
        logger.critical("Failed to load TOSAssistant: %s", e, exc_info=True)
        if settings.IS_PRODUCTION:
            # Hard exit so Cloud Run retries with a clean instance rather than
            # routing live traffic to a container with no working model.
            sys.exit(1)
        else:
            logger.warning("Continuing in dev mode without AI model.")