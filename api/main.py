import time
import uuid
import logging
from contextvars import ContextVar
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pythonjsonlogger import jsonlogger
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from starlette.middleware.base import BaseHTTPMiddleware

from api.core.config import settings
from api.core.database import init_db, init_storage, init_assistant
from api.core.security import limiter
from api.routers import auth, ingest, chat, history, admin

# ── Structured JSON logging ───────────────────────────────────────────────────
#
# WHY: Standard Python logging outputs unstructured text. In Cloud Run, you
# can't filter logs by request_id, user_id, or error type — you just get a wall
# of text. Structured JSON means every log line is a queryable record in
# Cloud Logging:
#
#   resource.labels.service = "tos-summarizer"
#   AND jsonPayload.request_id = "a3f9b2c1"
#   AND jsonPayload.level = "ERROR"
#
# This lets you find every log line for a specific failing request in seconds
# rather than grepping through thousands of lines.

# ContextVar holds the current request_id for the duration of each request.
# ContextVars are request-scoped in async frameworks — setting it in middleware
# makes it available to any logger called anywhere downstream in that request,
# including inside the RAG pipeline, without passing it as a function argument.
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class CorrelationJsonFormatter(jsonlogger.JsonFormatter):
    """
    Extends python-json-logger to inject the current request_id into every
    log record automatically. This is what links a Pinecone error log to the
    specific HTTP request that caused it.
    """
    def add_fields(self, log_record: dict, record: logging.LogRecord, message_dict: dict):
        super().add_fields(log_record, record, message_dict)
        log_record["request_id"] = _request_id_ctx.get()
        log_record["service"] = settings.PROJECT_NAME
        log_record["version"] = settings.VERSION
        # Rename levelname → level for Cloud Logging compatibility
        log_record["level"] = log_record.pop("levelname", record.levelname)


def _configure_logging():
    """
    Replace the default text handler with a JSON handler on the root logger.
    All loggers in the app (including third-party ones like uvicorn and
    langchain) inherit from root, so they all emit JSON automatically.
    """
    handler = logging.StreamHandler()
    formatter = CorrelationJsonFormatter(
        fmt="%(asctime)s %(level)s %(name)s %(message)s"
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)


_configure_logging()
logger = logging.getLogger(__name__)


# ── App lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    init_storage()
    init_assistant()
    logger.info("startup complete", extra={"event": "startup"})
    yield
    logger.info("shutdown", extra={"event": "shutdown"})


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    # Hide /docs in production — no need to expose your API schema publicly
    docs_url=None if settings.IS_PRODUCTION else "/docs",
    redoc_url=None if settings.IS_PRODUCTION else "/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Request tracing middleware ────────────────────────────────────────────────
#
# WHY: Every request gets a short unique ID (e.g. "a3f9b2c1"). It is:
#   1. Set on _request_id_ctx so every logger in this request emits it
#   2. Returned in the X-Request-ID response header so the frontend can
#      include it in bug reports ("my request ID was a3f9b2c1")
#   3. Logged with method, path, status, and duration so you have a
#      one-line audit trail for every request in Cloud Logging
#
# The ContextVar approach is important: it's async-safe. Unlike a threading
# local, a ContextVar is isolated per asyncio Task — two concurrent requests
# never see each other's request_id.
class RequestTracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = str(uuid.uuid4())[:8]

        # Store on ContextVar (for loggers) and request.state (for route handlers)
        token = _request_id_ctx.set(rid)
        request.state.request_id = rid

        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            # Always reset the ContextVar, even if the handler raises
            _request_id_ctx.reset(token)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": elapsed_ms,
            },
        )
        response.headers["X-Request-ID"] = rid
        return response


app.add_middleware(RequestTracingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(history.router)
app.include_router(admin.router)


@app.get("/health")
async def health():
    from api.core import database
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "model_ready": database.shared_assistant is not None,
    }