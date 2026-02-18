
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from database import SessionLocal
import models

router = APIRouter(prefix="/restaurants")

class RestaurantCreate(BaseModel):
    name: str
    phone: Optional[str] = None
    ivr_text: Optional[str] = None

class RestaurantUpdate(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    ivr_text: Optional[str] = None
    active: Optional[bool] = None

def serialize_restaurant(r: models.Restaurant):
    return {
        "id": r.id,
        "name": r.name,
        "phone": r.phone,
        "active": r.active,
        "ivr_text": r.ivr_text,
    }

@router.post("/add")
def add_restaurant(payload: RestaurantCreate):
    db = SessionLocal()
    try:
        r = models.Restaurant(
            name=payload.name.strip(),
            phone=(payload.phone or "").strip(),
            ivr_text=(payload.ivr_text or "").strip()
        )
        db.add(r)
        db.commit()
        db.refresh(r)
        return serialize_restaurant(r)
    finally:
        db.close()

@router.get("/")
def list_restaurants():
    db = SessionLocal()
    try:
        rows = db.query(models.Restaurant).all()
        return [serialize_restaurant(r) for r in rows]
    finally:
        db.close()

@router.post("/toggle/{id}")
def toggle(id:int):
    db = SessionLocal()
    try:
        r = db.get(models.Restaurant, id)
        if not r:
            raise HTTPException(status_code=404, detail="Restaurant not found")
        r.active = not r.active
        db.commit()
        return {"active": r.active}
    finally:
        db.close()

@router.put("/{id}")
def update_restaurant(id: int, payload: RestaurantUpdate):
    db = SessionLocal()
    try:
        r = db.get(models.Restaurant, id)
        if not r:
            raise HTTPException(status_code=404, detail="Restaurant not found")
        if payload.name is not None:
            r.name = payload.name.strip()
        if payload.phone is not None:
            r.phone = payload.phone.strip()
        if payload.ivr_text is not None:
            r.ivr_text = payload.ivr_text
        if payload.active is not None:
            r.active = payload.active
        db.commit()
        return serialize_restaurant(r)
    finally:
        db.close()

@router.delete("/{id}")
def delete_restaurant(id: int):
    db = SessionLocal()
    try:
        r = db.get(models.Restaurant, id)
        if not r:
            raise HTTPException(status_code=404, detail="Restaurant not found")
        db.delete(r)
        db.commit()
        return {"deleted": True}
    finally:
        db.close()
