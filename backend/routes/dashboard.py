
from fastapi import APIRouter, Depends
from database import SessionLocal
import models
from sqlalchemy import func
from auth import require_auth

router = APIRouter(prefix="/dashboard")

@router.get("/stats")
def stats(user=Depends(require_auth)):
    db = SessionLocal()
    try:
        if user.is_admin:
            total_restaurants = db.query(models.Restaurant).filter_by(active=True).count()
            total_calls = db.query(models.CallLog).count()
            missed = db.query(models.CallLog).filter_by(status="missed").count()
            total_duration = db.query(func.sum(models.CallLog.duration)).scalar()
            avg_duration = db.query(func.avg(models.CallLog.duration)).scalar()
        else:
            if not user.restaurant_id:
                return {
                    "restaurants": 0,
                    "calls": 0,
                    "missed": 0,
                    "total_duration": 0.0,
                    "avg_duration": 0.0,
                }
            r = db.get(models.Restaurant, user.restaurant_id)
            rname = r.name if r else None
            base = db.query(models.CallLog)
            if rname:
                base = base.filter(
                    (models.CallLog.restaurant_id == user.restaurant_id)
                    | ((models.CallLog.restaurant_id.is_(None)) & (models.CallLog.restaurant == rname))
                )
            else:
                base = base.filter(models.CallLog.restaurant_id == user.restaurant_id)
            total_restaurants = 1
            total_calls = base.count()
            missed = base.filter_by(status="missed").count()
            total_duration = base.with_entities(func.sum(models.CallLog.duration)).scalar()
            avg_duration = base.with_entities(func.avg(models.CallLog.duration)).scalar()
        return {
            "restaurants": total_restaurants,
            "calls": total_calls,
            "missed": missed,
            "total_duration": float(total_duration or 0),
            "avg_duration": float(avg_duration or 0),
        }
    finally:
        db.close()
