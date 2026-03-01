from fastapi import APIRouter, Depends
from database import SessionLocal
import models
from auth import require_auth

router = APIRouter(prefix="/orders")


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
        return [
            {
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
            for o in rows
        ]
    finally:
        db.close()
