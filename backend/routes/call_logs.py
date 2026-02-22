
from fastapi import APIRouter, Depends
from database import SessionLocal
import models
from auth import require_auth

router = APIRouter(prefix="/calls")

@router.get("/")
def logs(user=Depends(require_auth)):
    db = SessionLocal()
    try:
        q = db.query(models.CallLog)
        if not user.is_admin:
            if not user.restaurant_id:
                return []
            r = db.get(models.Restaurant, user.restaurant_id)
            rname = r.name if r else None
            if rname:
                q = q.filter(
                    (models.CallLog.restaurant_id == user.restaurant_id)
                    | ((models.CallLog.restaurant_id.is_(None)) & (models.CallLog.restaurant == rname))
                )
            else:
                q = q.filter(models.CallLog.restaurant_id == user.restaurant_id)
        rows = q.order_by(models.CallLog.id.desc()).all()
        return [
            {
                "id": l.id,
                "restaurant_id": l.restaurant_id,
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
