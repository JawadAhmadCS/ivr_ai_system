import json
from fastapi import APIRouter, Depends
from sqlalchemy import func
from database import SessionLocal
import models
from auth import require_auth

router = APIRouter(prefix="/calls")


def _usage_by_call_sid(db, call_sids: set[str]) -> dict[str, dict]:
    clean_sids = {str(s).strip() for s in (call_sids or set()) if str(s or "").strip()}
    if not clean_sids:
        return {}

    usage_map: dict[str, dict] = {}

    grouped = (
        db.query(
            models.ApiUsageLog.call_sid,
            func.sum(models.ApiUsageLog.total_tokens),
            func.sum(models.ApiUsageLog.cost_usd),
            func.count(models.ApiUsageLog.id),
        )
        .filter(models.ApiUsageLog.call_sid.in_(clean_sids))
        .group_by(models.ApiUsageLog.call_sid)
        .all()
    )
    for sid, total_tokens, total_cost, events in grouped:
        key = str(sid or "").strip()
        if not key:
            continue
        usage_map[key] = {
            "tokens": int(total_tokens or 0),
            "cost": float(total_cost or 0.0),
            "events": int(events or 0),
        }

    # Backfill older usage rows that only stored call_sid inside meta_json.
    legacy_rows = (
        db.query(
            models.ApiUsageLog.meta_json,
            models.ApiUsageLog.total_tokens,
            models.ApiUsageLog.cost_usd,
        )
        .filter(
            models.ApiUsageLog.call_sid.is_(None),
            models.ApiUsageLog.meta_json.isnot(None),
            models.ApiUsageLog.meta_json.like('%"call_sid"%'),
        )
        .all()
    )
    for meta_json, total_tokens, total_cost in legacy_rows:
        if not meta_json:
            continue
        try:
            payload = json.loads(meta_json)
        except Exception:
            continue
        sid = str((payload or {}).get("call_sid") or "").strip()
        if not sid or sid not in clean_sids:
            continue
        current = usage_map.setdefault(sid, {"tokens": 0, "cost": 0.0, "events": 0})
        current["tokens"] += int(total_tokens or 0)
        current["cost"] += float(total_cost or 0.0)
        current["events"] += 1

    return usage_map


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
        usage_map = _usage_by_call_sid(
            db,
            {str(l.call_sid).strip() for l in rows if str(l.call_sid or "").strip()},
        )
        out = []
        for l in rows:
            sid = str(l.call_sid or "").strip()
            usage = usage_map.get(sid) or {}
            out.append({
                "id": l.id,
                "restaurant_id": l.restaurant_id,
                "restaurant": l.restaurant,
                "caller": l.caller,
                "call_sid": l.call_sid,
                "duration": l.duration,
                "status": l.status,
                "api_usage_tokens": int(usage.get("tokens", 0)),
                "api_cost_usd": round(float(usage.get("cost", 0.0)), 8),
                "api_usage_events": int(usage.get("events", 0)),
                "created": l.created.isoformat() if l.created else None,
            })
        return out
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
