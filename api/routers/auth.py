import logging
from fastapi import APIRouter, HTTPException, Request
from api.core import database
from api.core.security import limiter
from api.schemas.auth import SignUpRequest, LoginRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/signup")
@limiter.limit("5/minute")
async def signup(request: Request, req: SignUpRequest):
    if not database.supa_admin:
        raise HTTPException(status_code=503, detail="Auth service not configured.")
    try:
        res = database.supa_admin.auth.sign_up({"email": req.email, "password": req.password})
        return {
            "message": "Account created. Check your email to confirm.",
            "user_id": res.user.id if res.user else None,
        }
    except Exception as exc:
        logger.warning("Signup failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

@router.post("/login")
@limiter.limit("10/minute")
async def login(request: Request, req: LoginRequest):
    if not database.supa_admin:
        raise HTTPException(status_code=503, detail="Auth service not configured.")
    try:
        res = database.supa_admin.auth.sign_in_with_password({"email": req.email, "password": req.password})
        return {
            "access_token": res.session.access_token,
            "refresh_token": res.session.refresh_token,
            "user": {
                "id": res.user.id,
                "email": res.user.email,
            }
        }
    except Exception as exc:
        logger.warning("Login failed for %s: %s", req.email, exc)
        raise HTTPException(status_code=401, detail="Invalid login credentials")
