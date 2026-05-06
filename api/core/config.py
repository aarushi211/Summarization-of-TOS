import os
import sys
from typing import List, Set, Any
from pathlib import Path
from pydantic import field_validator, model_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


def _default_model_path() -> Path:
    """
    Resolve MODEL_PATH relative to this file's location, not the process CWD.

    Problem: Path("models/...") resolves relative to wherever the user runs
    the process from. On desktop this is often wrong — e.g. a packaged app
    launched from /Applications/ would look for models/ in /Applications/.

    Fix: anchor the default to the project root (two levels up from this file:
    api/core/config.py → api/ → project_root/).
    This is stable regardless of CWD and works with PyInstaller's sys._MEIPASS.
    """
    # PyInstaller sets sys._MEIPASS to the unpacked bundle directory
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "models" / "legal_qwen.Q4_K_M.gguf"
    # Normal run: anchor to project root
    return Path(__file__).resolve().parent.parent.parent / "models" / "legal_qwen.Q4_K_M.gguf"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    PROJECT_NAME: str = "TOS Summarizer"
    VERSION: str = "2.0.0"

    API_DIR: Path = Path(__file__).resolve().parent.parent
    PROJECT_ROOT: Path = API_DIR.parent

    # ── Runtime mode ──────────────────────────────────────────────────────────
    # Set RUNTIME_MODE=desktop in the .env shipped with the desktop app.
    # Set RUNTIME_MODE=server (or leave unset) for Cloud Run.
    RUNTIME_MODE: str = "server"

    @property
    def is_desktop(self) -> bool:
        return self.RUNTIME_MODE.lower() == "desktop"

    @property
    def IS_PRODUCTION(self) -> bool:
        return not self.DEBUG and not self.is_desktop

    CLOUD_RUN_ENV: bool = False
    DEBUG: bool = False

    # ── Model ─────────────────────────────────────────────────────────────────
    # Default is resolved at class definition time so it's always absolute.
    MODEL_PATH: Path = Field(default_factory=_default_model_path)

    # ── Fix: GPU layers configurable ──────────────────────────────────────────
    # -1 = offload all layers to GPU (great for developers, crashes on
    # integrated graphics or low-VRAM machines).
    # 0  = CPU only (slow but runs anywhere).
    # Desktop .env can set N_GPU_LAYERS=0 for safety, or let power users set
    # it to a higher value for their hardware.
    N_GPU_LAYERS: int = -1

    # ── Desktop-only ──────────────────────────────────────────────────────────
    DESKTOP_DATA_DIR: Path = Path.home() / ".tos-summarizer"
    CHROMA_COLLECTION_NAME: str = "tos-documents"

    # ── Server-only ───────────────────────────────────────────────────────────
    PINECONE_API_KEY: str = Field(default="")
    PINECONE_INDEX_NAME: str = "tos-summarizer"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"
    PINECONE_DIMENSION: int = 768

    SUPABASE_URL: str = Field(default="")
    SUPABASE_SERVICE_KEY: str = Field(default="")

    AWS_S3_BUCKET_NAME: str = Field(default="")
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = Field(default="")
    AWS_SECRET_ACCESS_KEY: str = Field(default="")

    # Shared
    ALLOWED_ORIGINS: Any = Field(default="http://localhost:3000")
    ADMIN_SECRET: str = Field(default="")
    CLEANUP_DAYS: int = 30
    MAX_FILE_BYTES: int = 50 * 1024 * 1024
    ALLOWED_MIME_TYPES: Set[str] = {"application/pdf"}

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        if isinstance(v, list):
            return [str(item).strip() for item in v if item]
        return ["http://localhost:3000"]

    @field_validator("MODEL_PATH", mode="after")
    @classmethod
    def resolve_model_path(cls, v: Path) -> Path:
        # If an explicit path was provided via env var, make it absolute
        return v.absolute() if not v.is_absolute() else v

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> "Settings":
        is_test = (
            os.getenv("PYTEST_CURRENT_TEST")
            or os.getenv("CI")
            or "pytest" in sys.modules
        )
        if is_test:
            return self

        if self.is_desktop:
            self.DESKTOP_DATA_DIR.mkdir(parents=True, exist_ok=True)
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
                f"Missing required env vars for server mode: {missing}. "
                "Set them in .env, Cloud Run secrets, or set RUNTIME_MODE=desktop."
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
        settings = Settings.model_construct(
            PROJECT_NAME="TOS Summarizer Test",
            RUNTIME_MODE="server",
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
            MODEL_PATH=_default_model_path(),
            N_GPU_LAYERS=-1,
            DESKTOP_DATA_DIR=Path.home() / ".tos-summarizer-test",
            CHROMA_COLLECTION_NAME="tos-documents",
        )
    else:
        raise