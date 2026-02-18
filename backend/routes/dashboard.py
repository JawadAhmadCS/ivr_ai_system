
from fastapi import APIRouter
from database import SessionLocal
import models
from sqlalchemy import func

router = APIRouter(prefix="/dashboard")

@router.get("/stats")
def stats():
    db = SessionLocal()
    try:
        total_restaurants = db.query(models.Restaurant).filter_by(active=True).count()
        total_calls = db.query(models.CallLog).count()
        missed = db.query(models.CallLog).filter_by(status="missed").count()
        avg_duration = db.query(func.avg(models.CallLog.duration)).scalar()
        return {
            "restaurants": total_restaurants,
            "calls": total_calls,
            "missed": missed,
            "avg_duration": float(avg_duration or 0),
        }
    finally:
        db.close()
