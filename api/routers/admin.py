import logging
from fastapi import APIRouter, HTTPException, Query
from api.core.config import settings
from api.core import database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin"])

@router.post("/cleanup")
async def cleanup(secret: str = Query(...)):
    if not settings.ADMIN_SECRET or secret != settings.ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    
    # Logic to find old documents in Supabase and delete them
    # For now, just a placeholder as requested in the plan
    return {"message": "Cleanup job triggered (mock)"}
