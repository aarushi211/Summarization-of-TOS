import os
import sys
from typing import List, Set, Any, Optional
from pathlib import Path
from pydantic import field_validator, model_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    PROJECT_NAME: str = "TOS Summarizer"
    VERSION: str = "2.0.0"

    # Path setup
    API_DIR: Path = Path(__file__).resolve().parent.parent
    PROJECT_ROOT: Path = API_DIR.parent

    # Environment
    CLOUD_RUN_ENV: bool = False

    # ── FIX 1: DEBUG defaults to False. ──────────────────────────────────────
    # Previously was DEBUG: bool = True — if the env var was ever missing in a
    # deploy, production ran in debug mode, exposing tracebacks and disabling
    # security headers. Now you must explicitly set DEBUG=True in dev .env.
    DEBUG: bool = False

    @property
    def IS_PRODUCTION(self) -> bool:
        return self.CLOUD_RUN_ENV or not self.DEBUG

    # Model
    MODEL_PATH: Path = Field(default=Path("models/legal_qwen.Q4_K_M.gguf"))

    # Pinecone
    PINECONE_API_KEY: str = Field(...)
    PINECONE_INDEX_NAME: str = "tos-summarizer"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"
    PINECONE_DIMENSION: int = 768

    # Auth & DB
    SUPABASE_URL: str = Field(...)
    SUPABASE_SERVICE_KEY: str = Field(...)

    # Storage
    AWS_S3_BUCKET_NAME: str = Field(...)
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = Field(...)
    AWS_SECRET_ACCESS_KEY: str = Field(...)

    # Security
    ALLOWED_ORIGINS: Any = Field(...)
    ADMIN_SECRET: str = Field(...)
    CLEANUP_DAYS: int = 30

    # Limits
    MAX_FILE_BYTES: int = 50 * 1024 * 1024
    ALLOWED_MIME_TYPES: Set[str] = {"application/pdf"}

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        if isinstance(v, list):
            return [str(item).strip() for item in v if item]
        return []

    @field_validator("MODEL_PATH", mode="after")
    @classmethod
    def resolve_model_path(cls, v: Path) -> Path:
        if not v.is_absolute():
            return v.absolute()
        return v

    # ── FIX 2: Fail-fast startup validation. ─────────────────────────────────
    # Previously, missing env vars silently became empty strings and only
    # failed at runtime during the first request. Now we check at import time
    # so Cloud Run startup fails loudly before serving any traffic.
    @model_validator(mode="after")
    def validate_production_requirements(self) -> "Settings":
        is_test = (
            os.getenv("PYTEST_CURRENT_TEST")
            or os.getenv("CI")
            or "pytest" in sys.modules
        )
        if is_test:
            return self

        required = {
            "PINECONE_API_KEY": self.PINECONE_API_KEY,
            "SUPABASE_URL": self.SUPABASE_URL,
            "SUPABASE_SERVICE_KEY": self.SUPABASE_SERVICE_KEY,
            "AWS_S3_BUCKET_NAME": self.AWS_S3_BUCKET_NAME,
            "AWS_ACCESS_KEY_ID": self.AWS_ACCESS_KEY_ID,
            "AWS_SECRET_ACCESS_KEY": self.AWS_SECRET_ACCESS_KEY,
            "ADMIN_SECRET": self.ADMIN_SECRET,
        }
        missing = [k for k, v in required.items() if not v or v in ("test", "test-key")]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {missing}. "
                "Set them in .env or Cloud Run secrets before starting."
            )
        return self


try:
    settings = Settings()
except Exception as e:
    if (
        os.getenv("PYTEST_CURRENT_TEST")
        or os.getenv("CI")
        or "pytest" in sys.modules
    ):
        # Provide dummy settings so imports don't fail during test collection.
        # The actual tests patch these values anyway.
        settings = Settings.model_construct(
            PROJECT_NAME="TOS Summarizer Test",
            DEBUG=True,
            CLOUD_RUN_ENV=False,
            SUPABASE_URL="https://test.supabase.co",
            SUPABASE_SERVICE_KEY="test-key",
            AWS_S3_BUCKET_NAME="test-bucket",
            AWS_ACCESS_KEY_ID="test",
            AWS_SECRET_ACCESS_KEY="test",
            ALLOWED_ORIGINS=["http://localhost:3000"],
            ADMIN_SECRET="test",
            PINECONE_API_KEY="test",
            PINECONE_INDEX_NAME="tos-summarizer",
            PINECONE_CLOUD="aws",
            PINECONE_REGION="us-east-1",
            PINECONE_DIMENSION=768,
            AWS_REGION="us-east-1",
            MAX_FILE_BYTES=50 * 1024 * 1024,
            ALLOWED_MIME_TYPES={"application/pdf"},
            MODEL_PATH=Path("models/legal_qwen.Q4_K_M.gguf"),
        )
    else:
        raise