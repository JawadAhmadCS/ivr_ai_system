
from fastapi import APIRouter
from database import SessionLocal
import models

router = APIRouter(prefix="/calls")

@router.get("/")
def logs():
    db = SessionLocal()
    return db.query(models.CallLog).all()
