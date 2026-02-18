
from fastapi import APIRouter
from database import SessionLocal
import models

router = APIRouter(prefix="/calls")

@router.get("/")
def logs():
    db = SessionLocal()
    try:
        rows = db.query(models.CallLog).order_by(models.CallLog.id.desc()).all()
        return [
            {
                "id": l.id,
                "restaurant": l.restaurant,
                "caller": l.caller,
                "duration": l.duration,
                "status": l.status,
                "created": l.created.isoformat() if l.created else None,
            }
            for l in rows
        ]
    finally:
        db.close()
