
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from database import SessionLocal
import models
from auth import require_auth, require_admin

router = APIRouter(prefix="/restaurants")
RECORDINGS_DIR = (Path(__file__).resolve().parent.parent / "recordings").resolve()
MAX_OPENING_AUDIO_BYTES = 10 * 1024 * 1024
ALLOWED_OPENING_AUDIO_EXTENSIONS = {".wav", ".mp3"}

class RestaurantCreate(BaseModel):
    name: str
    phone: Optional[str] = None
    ivr_text: Optional[str] = None
    precall_notice_text: Optional[str] = None
    precall_notice_audio_url: Optional[str] = None

class RestaurantUpdate(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    ivr_text: Optional[str] = None
    precall_notice_text: Optional[str] = None
    precall_notice_audio_url: Optional[str] = None
    active: Optional[bool] = None

def serialize_restaurant(r: models.Restaurant):
    return {
        "id": r.id,
        "name": r.name,
        "phone": r.phone,
        "active": r.active,
        "ivr_text": r.ivr_text,
        "precall_notice_text": r.precall_notice_text,
        "precall_notice_audio_url": r.precall_notice_audio_url,
    }


def _can_access_restaurant(user, restaurant_id: int) -> bool:
    if user.is_admin:
        return True
    return bool(user.restaurant_id and user.restaurant_id == restaurant_id)


def _delete_local_recording_file(recording_url: str | None) -> bool:
    if not recording_url:
        return False
    raw = str(recording_url).strip()
    if not raw:
        return False

    parsed_path = urlparse(raw).path or raw
    if not (
        parsed_path.startswith("/recordings/")
        or parsed_path.startswith("/api/recordings/")
    ):
        return False

    filename = Path(parsed_path).name
    if not filename:
        return False

    target = (RECORDINGS_DIR / filename).resolve()
    if not str(target).startswith(str(RECORDINGS_DIR)):
        return False
    if not target.exists() or not target.is_file():
        return False

    target.unlink()
    return True

@router.post("/add")
def add_restaurant(payload: RestaurantCreate, _user=Depends(require_admin)):
    db = SessionLocal()
    try:
        r = models.Restaurant(
            name=payload.name.strip(),
            phone=(payload.phone or "").strip(),
            ivr_text=(payload.ivr_text or "").strip(),
            precall_notice_text=(payload.precall_notice_text or "").strip(),
            precall_notice_audio_url=(payload.precall_notice_audio_url or "").strip() or None,
        )
        db.add(r)
        db.commit()
        db.refresh(r)
        return serialize_restaurant(r)
    finally:
        db.close()

@router.get("/")
def list_restaurants(user=Depends(require_auth)):
    db = SessionLocal()
    try:
        q = db.query(models.Restaurant)
        if not user.is_admin:
            if not user.restaurant_id:
                return []
            q = q.filter_by(id=user.restaurant_id)
        rows = q.all()
        return [serialize_restaurant(r) for r in rows]
    finally:
        db.close()

@router.post("/toggle/{id}")
def toggle(id:int, _user=Depends(require_admin)):
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
def update_restaurant(id: int, payload: RestaurantUpdate, user=Depends(require_auth)):
    db = SessionLocal()
    try:
        r = db.get(models.Restaurant, id)
        if not r:
            raise HTTPException(status_code=404, detail="Restaurant not found")
        if not _can_access_restaurant(user, id):
            raise HTTPException(status_code=403, detail="Forbidden")
        if payload.name is not None:
            r.name = payload.name.strip()
        if payload.phone is not None:
            r.phone = payload.phone.strip()
        if payload.ivr_text is not None:
            r.ivr_text = payload.ivr_text
        if payload.precall_notice_text is not None:
            r.precall_notice_text = payload.precall_notice_text
        if payload.precall_notice_audio_url is not None:
            r.precall_notice_audio_url = (payload.precall_notice_audio_url or "").strip() or None
        if payload.active is not None:
            if not user.is_admin:
                raise HTTPException(status_code=403, detail="Forbidden")
            r.active = payload.active
        db.commit()
        return serialize_restaurant(r)
    finally:
        db.close()


@router.post("/{id}/opening-audio")
async def upload_opening_audio(
    id: int,
    file: UploadFile = File(...),
    user=Depends(require_auth),
):
    db = SessionLocal()
    target_path: Path | None = None
    try:
        r = db.get(models.Restaurant, id)
        if not r:
            raise HTTPException(status_code=404, detail="Restaurant not found")
        if not _can_access_restaurant(user, id):
            raise HTTPException(status_code=403, detail="Forbidden")

        original_name = str(file.filename or "").strip()
        ext = Path(original_name).suffix.lower()
        if ext not in ALLOWED_OPENING_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Unsupported audio format. Allowed: wav, mp3",
            )

        payload = await file.read(MAX_OPENING_AUDIO_BYTES + 1)
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(payload) > MAX_OPENING_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="File too large (max 10 MB)")

        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"opening_{id}_{uuid4().hex}{ext}"
        target_path = (RECORDINGS_DIR / filename).resolve()
        if not str(target_path).startswith(str(RECORDINGS_DIR)):
            raise HTTPException(status_code=400, detail="Invalid file path")
        target_path.write_bytes(payload)

        previous_url = r.precall_notice_audio_url
        r.precall_notice_audio_url = f"/api/recordings/{filename}"
        db.commit()
        db.refresh(r)

        if previous_url and previous_url != r.precall_notice_audio_url:
            _delete_local_recording_file(previous_url)

        return {
            "ok": True,
            "restaurant": serialize_restaurant(r),
            "precall_notice_audio_url": r.precall_notice_audio_url,
        }
    except HTTPException:
        if target_path and target_path.exists():
            try:
                target_path.unlink()
            except Exception:
                pass
        raise
    except Exception:
        if target_path and target_path.exists():
            try:
                target_path.unlink()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail="Failed to upload opening audio")
    finally:
        try:
            await file.close()
        except Exception:
            pass
        db.close()


@router.delete("/{id}/opening-audio")
def delete_opening_audio(id: int, user=Depends(require_auth)):
    db = SessionLocal()
    try:
        r = db.get(models.Restaurant, id)
        if not r:
            raise HTTPException(status_code=404, detail="Restaurant not found")
        if not _can_access_restaurant(user, id):
            raise HTTPException(status_code=403, detail="Forbidden")

        file_deleted = _delete_local_recording_file(r.precall_notice_audio_url)
        r.precall_notice_audio_url = None
        db.commit()
        db.refresh(r)
        return {"ok": True, "file_deleted": file_deleted, "restaurant": serialize_restaurant(r)}
    finally:
        db.close()


@router.delete("/{id}")
def delete_restaurant(id: int, _user=Depends(require_admin)):
    db = SessionLocal()
    try:
        r = db.get(models.Restaurant, id)
        if not r:
            raise HTTPException(status_code=404, detail="Restaurant not found")
        _delete_local_recording_file(r.precall_notice_audio_url)
        db.delete(r)
        db.commit()
        return {"deleted": True}
    finally:
        db.close()
