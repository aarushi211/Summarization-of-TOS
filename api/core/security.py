import logging
from typing import Optional
from fastapi import Header, HTTPException, Request, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address
from api.core import database

logger = logging.getLogger(__name__)

# Rate limiter setup
def _rate_limit_key(request: Request) -> str:
    uid = getattr(request.state, "user_id", None)
    return uid if uid else get_remote_address(request)

limiter = Limiter(key_func=_rate_limit_key)

async def get_current_user(authorization: Optional[str] = Header(default=None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")

    token = authorization.split(" ")[1]
    
    if not database.supa_admin:
        raise HTTPException(status_code=503, detail="Database not configured.")

    try:
        # Ask Supabase directly if the token is valid
        res = database.supa_admin.auth.get_user(token)
        if not res.user:
            raise HTTPException(status_code=401, detail="Token rejected by Supabase.")
            
        return {"user_id": res.user.id, "token": token}

    except Exception as exc:
        logger.warning("JWT validation failed via Supabase: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid token.")
