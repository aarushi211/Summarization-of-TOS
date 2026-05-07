import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pythonjsonlogger import jsonlogger
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

from api.core.config import settings
from api.core.database import init_db, init_storage, init_assistant
from api.core.security import limiter
from api.routers import admin, auth, chat, history, ingest

# ── Structured JSON logging ───────────────────────────────────────────────────
#
# WHY: Standard Python logging outputs unstructured text. In Cloud Run, you
# can't filter logs by request_id, user_id, or error type — you just get a wall
# of text. Structured JSON means every log line is a queryable record in
# Cloud Logging.

_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class CorrelationJsonFormatter(jsonlogger.JsonFormatter):
    """Inject request metadata into every log record."""

    def add_fields(self, log_record: dict, record: logging.LogRecord, message_dict: dict):
        super().add_fields(log_record, record, message_dict)
        log_record["request_id"] = _request_id_ctx.get()
        log_record["service"] = settings.PROJECT_NAME
        log_record["version"] = settings.VERSION
        log_record["level"] = log_record.pop("levelname", record.levelname)


def _configure_logging() -> None:
    """Configure root logging to emit JSON."""
    handler = logging.StreamHandler()
    formatter = CorrelationJsonFormatter(fmt="%(asctime)s %(level)s %(name)s %(message)s")
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)


_configure_logging()
logger = logging.getLogger(__name__)


async def _warm_assistant() -> None:
    """
    Load the assistant after startup so Cloud Run can see the server listening
    on PORT before any expensive model initialization finishes.

    If init_assistant() is synchronous/heavy, run it in a worker thread.
    """
    try:
        await asyncio.to_thread(init_assistant)
        logger.info("assistant initialized", extra={"event": "assistant_ready"})
    except Exception:
        logger.exception("assistant initialization failed", extra={"event": "assistant_init_failed"})


# ── App lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Keep startup as light as possible so the container can bind to PORT fast.
    try:
        await asyncio.to_thread(init_db)
        await asyncio.to_thread(init_storage)
        logger.info("core startup complete", extra={"event": "startup_core_complete"})
    except Exception:
        logger.exception("core startup failed", extra={"event": "startup_core_failed"})
        raise

    # Fire-and-forget assistant warmup after the app has started.
    app.state.assistant_warmup_task = asyncio.create_task(_warm_assistant())

    yield

    # Best-effort shutdown cleanup.
    warmup_task = getattr(app.state, "assistant_warmup_task", None)
    if warmup_task and not warmup_task.done():
        warmup_task.cancel()
        try:
            await warmup_task
        except asyncio.CancelledError:
            pass

    logger.info("shutdown", extra={"event": "shutdown"})


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url=None if settings.IS_PRODUCTION else "/docs",
    redoc_url=None if settings.IS_PRODUCTION else "/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Request tracing middleware ────────────────────────────────────────────────
class RequestTracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = str(uuid.uuid4())[:8]
        token = _request_id_ctx.set(rid)
        request.state.request_id = rid

        t0 = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            _request_id_ctx.reset(token)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

            if response is not None:
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
            else:
                logger.warning(
                    "request_failed_before_response",
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "duration_ms": elapsed_ms,
                    },
                )


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
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "debug": settings.DEBUG,
    }
