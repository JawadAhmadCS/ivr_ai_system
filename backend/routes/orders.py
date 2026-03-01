from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from database import SessionLocal
import models
from auth import require_auth

router = APIRouter(prefix="/orders")


class OrderUpdatePayload(BaseModel):
    full_name: str | None = None
    order_type: str | None = None
    date_arrival: str | None = None
    time_arrival: str | None = None
    total_peoples: str | int | None = None
    contact_number: str | None = None
    status: str | None = None


def _serialize_order(o: models.Order) -> dict:
    return {
        "id": o.id,
        "restaurant_id": o.restaurant_id,
        "restaurant": o.restaurant,
        "caller": o.caller,
        "call_sid": o.call_sid,
        "order_type": o.order_type,
        "full_name": o.full_name,
        "address": o.address,
        "house_number": o.house_number,
        "ordered_items": o.ordered_items,
        "payment_method": o.payment_method,
        "status": o.status,
        "recording_sid": o.recording_sid,
        "recording_url": o.recording_url,
        "created": o.created.isoformat() if o.created else None,
    }


def _can_access_order(db, user, order: models.Order) -> bool:
    if user.is_admin:
        return True
    if not user.restaurant_id:
        return False
    if order.restaurant_id == user.restaurant_id:
        return True
    if order.restaurant_id is not None:
        return False
    r = db.get(models.Restaurant, user.restaurant_id)
    rname = r.name if r else None
    return bool(rname and order.restaurant == rname)


@router.get("/")
def list_orders(user=Depends(require_auth)):
    db = SessionLocal()
    try:
        q = db.query(models.Order)
        if not user.is_admin:
            if not user.restaurant_id:
                return []
            r = db.get(models.Restaurant, user.restaurant_id)
            rname = r.name if r else None
            if rname:
                q = q.filter(
                    (models.Order.restaurant_id == user.restaurant_id)
                    | ((models.Order.restaurant_id.is_(None)) & (models.Order.restaurant == rname))
                )
            else:
                q = q.filter(models.Order.restaurant_id == user.restaurant_id)
        rows = q.order_by(models.Order.id.desc()).all()
        return [_serialize_order(o) for o in rows]
    finally:
        db.close()


@router.put("/{order_id}")
def update_order(order_id: int, payload: OrderUpdatePayload, user=Depends(require_auth)):
    db = SessionLocal()
    try:
        row = db.get(models.Order, order_id)
        if not row:
            raise HTTPException(status_code=404, detail="Order not found")
        if not _can_access_order(db, user, row):
            raise HTTPException(status_code=403, detail="Forbidden")

        updated = False
        if payload.full_name is not None:
            row.full_name = str(payload.full_name).strip()
            updated = True
        if payload.order_type is not None:
            row.order_type = str(payload.order_type).strip()
            updated = True
        if payload.date_arrival is not None:
            row.address = str(payload.date_arrival).strip()
            updated = True
        if payload.time_arrival is not None:
            row.house_number = str(payload.time_arrival).strip()
            updated = True
        if payload.total_peoples is not None:
            row.ordered_items = str(payload.total_peoples).strip()
            updated = True
        if payload.contact_number is not None:
            row.payment_method = str(payload.contact_number).strip()
            updated = True
        if payload.status is not None:
            row.status = str(payload.status).strip()
            updated = True

        if not updated:
            raise HTTPException(status_code=400, detail="Nothing to update")

        db.commit()
        db.refresh(row)
        return {"ok": True, "order": _serialize_order(row)}
    finally:
        db.close()


@router.delete("/{order_id}")
def delete_order(order_id: int, user=Depends(require_auth)):
    db = SessionLocal()
    try:
        row = db.get(models.Order, order_id)
        if not row:
            raise HTTPException(status_code=404, detail="Order not found")
        if not _can_access_order(db, user, row):
            raise HTTPException(status_code=403, detail="Forbidden")
        db.delete(row)
        db.commit()
        return {"ok": True}
    finally:
        db.close()
