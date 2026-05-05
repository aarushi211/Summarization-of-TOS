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
    global shared_assistant
    logger.info("Loading TOSAssistant from %s...", settings.MODEL_PATH)
    if settings.MODEL_PATH.exists():
        try:
            shared_assistant = TOSAssistant(
                model_path=str(settings.MODEL_PATH),
                pinecone_api_key=settings.PINECONE_API_KEY,
                index_name=settings.PINECONE_INDEX_NAME,
                dimension=settings.PINECONE_DIMENSION,
                cloud=settings.PINECONE_CLOUD,
                region=settings.PINECONE_REGION
            )
            logger.info("TOSAssistant loaded successfully.")
        except Exception as e:
            logger.error("Failed to load TOSAssistant: %s", e)
    else:
        logger.warning("Model file not found at %s. AI features will fail.", settings.MODEL_PATH)
