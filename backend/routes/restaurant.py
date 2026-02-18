
from fastapi import APIRouter
from database import SessionLocal
import models

router = APIRouter(prefix="/restaurants")

@router.post("/add")
def add_restaurant(name:str, phone:str):
    db = SessionLocal()
    r = models.Restaurant(name=name, phone=phone)
    db.add(r)
    db.commit()
    return {"msg":"restaurant added"}

@router.get("/")
def list_restaurants():
    db = SessionLocal()
    return db.query(models.Restaurant).all()

@router.post("/toggle/{id}")
def toggle(id:int):
    db = SessionLocal()
    r = db.query(models.Restaurant).get(id)
    r.active = not r.active
    db.commit()
    return {"active": r.active}
