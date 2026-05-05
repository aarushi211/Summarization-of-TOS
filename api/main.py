import time
import uuid
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from starlette.middleware.base import BaseHTTPMiddleware

from api.core.config import settings
from api.core.database import init_db, init_s3, init_assistant
from api.core.security import limiter
from api.routers import auth, ingest, chat, history, admin

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    init_s3()
    init_assistant()
    yield
    logger.info("Shutting down.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url=None if settings.IS_PRODUCTION else "/docs",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class RequestTracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = str(uuid.uuid4())[:8]
        request.state.request_id = rid
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - t0
        logger.info(
            "request completed",
            extra={
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(elapsed * 1000, 1),
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

# Register Routers
app.include_router(auth.router)
app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(history.router)
app.include_router(admin.router)

@app.get("/health")
async def health():
    return {"status": "healthy", "version": settings.VERSION}