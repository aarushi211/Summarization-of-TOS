import os
from typing import List, Set, Any, Optional
from pathlib import Path
from pydantic import field_validator, Field
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
    
    @property
    def IS_PRODUCTION(self) -> bool:
        return self.CLOUD_RUN_ENV

    DEBUG: bool = True
    
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
            # In a real app we'd use PROJECT_ROOT
            return v.absolute()
        return v

try:
    settings = Settings()
except Exception as e:
    if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("CI"):
        # Provide dummy settings so imports don't fail during test collection
        # The actual tests will patch these anyway
        settings = Settings.model_construct(
            PROJECT_NAME="TOS Summarizer Test",
            SUPABASE_URL="https://test.supabase.co",
            SUPABASE_SERVICE_KEY="test-key",
            AWS_S3_BUCKET_NAME="test-bucket",
            AWS_ACCESS_KEY_ID="test",
            AWS_SECRET_ACCESS_KEY="test",
            ALLOWED_ORIGINS=["http://localhost:3000"],
            ADMIN_SECRET="test",
            PINECONE_API_KEY="test"
        )
    else:
        raise e
