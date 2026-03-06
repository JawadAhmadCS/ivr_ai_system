
from fastapi import APIRouter, Depends
from database import SessionLocal
import models
from auth import require_auth

router = APIRouter(prefix="/calls")


def _can_access_call(db, user, call: models.CallLog) -> bool:
    if user.is_admin:
        return True
    if not user.restaurant_id:
        return False
    if call.restaurant_id == user.restaurant_id:
        return True
    if call.restaurant_id is not None:
        return False
    r = db.get(models.Restaurant, user.restaurant_id)
    rname = r.name if r else None
    return bool(rname and call.restaurant == rname)


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
                "call_sid": l.call_sid,
                "duration": l.duration,
                "status": l.status,
                "created": l.created.isoformat() if l.created else None,
            }
            for l in rows
        ]
    finally:
        db.close()


@router.get("/{call_id}/transcript")
def transcript(call_id: int, user=Depends(require_auth)):
    db = SessionLocal()
    try:
        call = db.get(models.CallLog, call_id)
        if not call:
            return {"call": None, "items": []}
        if not _can_access_call(db, user, call):
            return {"call": None, "items": []}

        sid = str(call.call_sid or "").strip()
        if not sid:
            return {
                "call": {
                    "id": call.id,
                    "restaurant": call.restaurant,
                    "caller": call.caller,
                    "created": call.created.isoformat() if call.created else None,
                },
                "items": [],
            }

        rows = (
            db.query(models.CallTranscript)
            .filter(models.CallTranscript.call_sid == sid)
            .order_by(models.CallTranscript.seq.asc(), models.CallTranscript.id.asc())
            .all()
        )
        return {
            "call": {
                "id": call.id,
                "restaurant": call.restaurant,
                "caller": call.caller,
                "created": call.created.isoformat() if call.created else None,
            },
            "items": [
                {
                    "id": r.id,
                    "speaker": r.speaker,
                    "content": r.content,
                    "created": r.created.isoformat() if r.created else None,
                }
                for r in rows
            ],
        }
    finally:
        db.close()
