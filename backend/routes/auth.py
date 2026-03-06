import os
import secrets
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from database import SessionLocal
import models
from auth import (
    verify_password,
    create_token,
    ensure_admin_user,
    require_auth,
    require_admin,
    require_token,
    hash_password,
    revoke_user_tokens,
)

router = APIRouter(prefix="/auth")


class LoginPayload(BaseModel):
    email: str
    password: str


class UserCreatePayload(BaseModel):
    email: str
    password: str
    restaurant_id: int | None = None
    is_admin: bool | None = None


class UserAssignPayload(BaseModel):
    email: str
    restaurant_id: int | None = None


class UserStatusPayload(BaseModel):
    active: bool


class UserPasswordPayload(BaseModel):
    password: str


class UserUpdatePayload(BaseModel):
    email: str | None = None
    password: str | None = None


class ProfileUpdatePayload(BaseModel):
    name: str | None = None


class ProfilePasswordPayload(BaseModel):
    current_password: str
    new_password: str


class RecoveryKeysUnlockPayload(BaseModel):
    current_password: str


class SuperAdminRecoveryPayload(BaseModel):
    email: str
    recovery_key: str
    new_password: str


RECOVERY_KEY_COUNT = 10


def _configured_admin_email() -> str:
    return (os.getenv("ADMIN_EMAIL") or "").strip().lower()


def _is_owner_user(user: models.User | None) -> bool:
    if not user or not user.is_admin:
        return False
    configured = _configured_admin_email()
    if configured:
        return str(user.email or "").strip().lower() == configured
    return True


def _require_owner(user=Depends(require_auth)):
    if not _is_owner_user(user):
        raise HTTPException(status_code=403, detail="Owner access required")
    return user


def _new_recovery_key_value() -> str:
    raw = secrets.token_hex(16).upper()
    return "-".join(raw[i:i + 8] for i in range(0, 32, 8))


def _create_recovery_keys(db, user_id: int, count: int = RECOVERY_KEY_COUNT) -> list[models.RecoveryKey]:
    rows: list[models.RecoveryKey] = []
    for _ in range(max(1, int(count or 0))):
        key_value = None
        for _attempt in range(20):
            candidate = _new_recovery_key_value()
            exists = db.query(models.RecoveryKey.id).filter_by(key_value=candidate).first()
            if not exists:
                key_value = candidate
                break
        if not key_value:
            raise HTTPException(status_code=500, detail="Failed to generate unique recovery key")
        row = models.RecoveryKey(user_id=user_id, key_value=key_value, is_used=False)
        db.add(row)
        rows.append(row)
    db.commit()
    for row in rows:
        db.refresh(row)
    return rows


def _serialize_recovery_key(row: models.RecoveryKey) -> dict:
    return {
        "id": row.id,
        "key": row.key_value,
        "is_used": bool(row.is_used),
        "used_at": row.used_at.isoformat() if row.used_at else None,
        "created": row.created.isoformat() if row.created else None,
    }


@router.post("/login")
def login(payload: LoginPayload):
    ensure_admin_user()
    db = SessionLocal()
    try:
        email = payload.email.strip().lower()
        if "@" not in email or "." not in email:
            raise HTTPException(status_code=400, detail="Invalid email")
        user = db.query(models.User).filter_by(email=email).first()
        if not user or not user.active:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if not verify_password(payload.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
    finally:
        db.close()

    token = create_token(user.id)
    return {"token": token.token, "user": {"email": user.email}}


@router.post("/recover-super-admin")
def recover_super_admin(payload: SuperAdminRecoveryPayload):
    ensure_admin_user()
    email = (payload.email or "").strip().lower()
    recovery_key = (payload.recovery_key or "").strip()
    new_password = (payload.new_password or "").strip()

    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Invalid email")
    if not recovery_key:
        raise HTTPException(status_code=400, detail="Recovery key is required")
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="Password too short")

    configured_admin_email = _configured_admin_email()
    if configured_admin_email and email != configured_admin_email:
        raise HTTPException(status_code=403, detail="Recovery allowed only for super admin email")

    db = SessionLocal()
    try:
        user = db.query(models.User).filter_by(email=email).first()
        if user and not user.is_admin:
            raise HTTPException(status_code=403, detail="Recovery allowed only for admin accounts")

        expected_key = (os.getenv("SUPERADMIN_RECOVERY_KEY") or "").strip()
        static_key_ok = bool(expected_key) and secrets.compare_digest(recovery_key, expected_key)
        db_key_row = None
        has_db_keys = False

        owner_for_key = user
        if not owner_for_key and configured_admin_email:
            owner_for_key = db.query(models.User).filter_by(email=configured_admin_email).first()

        if owner_for_key:
            has_db_keys = bool(
                db.query(models.RecoveryKey.id).filter_by(user_id=owner_for_key.id).first()
            )
            db_key_row = (
                db.query(models.RecoveryKey)
                .filter_by(
                    user_id=owner_for_key.id,
                    key_value=recovery_key,
                    is_used=False,
                )
                .first()
            )

        if not static_key_ok and not db_key_row:
            if expected_key or has_db_keys:
                raise HTTPException(status_code=401, detail="Invalid recovery key")
            raise HTTPException(status_code=503, detail="Recovery is not configured on server")

        if db_key_row:
            db_key_row.is_used = True
            db_key_row.used_at = datetime.utcnow()

        if not user:
            user = models.User(
                email=email,
                password_hash=hash_password(new_password),
                active=True,
                is_admin=True,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        else:
            user.password_hash = hash_password(new_password)
            user.active = True
            user.is_admin = True
            db.commit()

        revoke_user_tokens(user.id)
        return {"ok": True, "email": user.email}
    finally:
        db.close()


@router.post("/logout")
def logout(token_value: str = Depends(require_token)):
    from auth import revoke_token

    revoke_token(token_value)
    return {"ok": True}


@router.get("/me")
def me(user=Depends(require_auth)):
    return {
        "email": user.email,
        "name": user.name,
        "id": user.id,
        "is_admin": user.is_admin,
        "is_owner": _is_owner_user(user),
        "restaurant_id": user.restaurant_id,
    }


@router.get("/profile")
def profile(user=Depends(require_auth)):
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "is_admin": user.is_admin,
        "is_owner": _is_owner_user(user),
    }


@router.put("/profile")
def update_profile(payload: ProfileUpdatePayload, user=Depends(require_auth)):
    new_name = (payload.name or "").strip()
    if len(new_name) > 120:
        raise HTTPException(status_code=400, detail="Name too long")
    db = SessionLocal()
    try:
        row = db.get(models.User, user.id)
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        row.name = new_name or None
        db.commit()
        db.refresh(row)
        return {"ok": True, "name": row.name}
    finally:
        db.close()


@router.put("/profile/password")
def update_profile_password(payload: ProfilePasswordPayload, user=Depends(require_auth)):
    current_password = (payload.current_password or "").strip()
    new_password = (payload.new_password or "").strip()
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="Password too short")

    db = SessionLocal()
    try:
        row = db.get(models.User, user.id)
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        if not verify_password(current_password, row.password_hash):
            raise HTTPException(status_code=401, detail="Current password is incorrect")
        row.password_hash = hash_password(new_password)
        db.commit()
        revoke_user_tokens(row.id)
        return {"ok": True, "force_relogin": True}
    finally:
        db.close()


@router.post("/profile/recovery-keys/unlock")
def unlock_recovery_keys(payload: RecoveryKeysUnlockPayload, owner=Depends(_require_owner)):
    current_password = (payload.current_password or "").strip()
    if not current_password:
        raise HTTPException(status_code=400, detail="Current password is required")

    db = SessionLocal()
    try:
        owner_row = db.get(models.User, owner.id)
        if not owner_row:
            raise HTTPException(status_code=404, detail="User not found")
        if not verify_password(current_password, owner_row.password_hash):
            raise HTTPException(status_code=401, detail="Current password is incorrect")

        rows = (
            db.query(models.RecoveryKey)
            .filter_by(user_id=owner.id)
            .order_by(models.RecoveryKey.id.asc())
            .all()
        )
        if not rows:
            rows = _create_recovery_keys(db, owner.id, RECOVERY_KEY_COUNT)
        return {
            "owner_email": owner.email,
            "total": len(rows),
            "keys": [_serialize_recovery_key(r) for r in rows],
        }
    finally:
        db.close()


@router.post("/profile/recovery-keys/regenerate")
def regenerate_recovery_keys(payload: RecoveryKeysUnlockPayload, owner=Depends(_require_owner)):
    current_password = (payload.current_password or "").strip()
    if not current_password:
        raise HTTPException(status_code=400, detail="Current password is required")

    db = SessionLocal()
    try:
        owner_row = db.get(models.User, owner.id)
        if not owner_row:
            raise HTTPException(status_code=404, detail="User not found")
        if not verify_password(current_password, owner_row.password_hash):
            raise HTTPException(status_code=401, detail="Current password is incorrect")

        db.query(models.RecoveryKey).filter_by(user_id=owner.id).delete(synchronize_session=False)
        db.commit()
        rows = _create_recovery_keys(db, owner.id, RECOVERY_KEY_COUNT)
        return {
            "owner_email": owner.email,
            "total": len(rows),
            "keys": [_serialize_recovery_key(r) for r in rows],
        }
    finally:
        db.close()


@router.get("/users")
def list_users(_user=Depends(require_admin)):
    db = SessionLocal()
    try:
        rows = db.query(models.User).order_by(models.User.id.desc()).all()
        return [
            {
                "id": u.id,
                "email": u.email,
                "active": u.active,
                "is_admin": u.is_admin,
                "restaurant_id": u.restaurant_id,
            }
            for u in rows
        ]
    finally:
        db.close()


@router.post("/users")
def create_user(payload: UserCreatePayload, _user=Depends(require_admin)):
    email = payload.email.strip().lower()
    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Invalid email")
    if len(payload.password or "") < 6:
        raise HTTPException(status_code=400, detail="Password too short")
    db = SessionLocal()
    try:
        existing = db.query(models.User).filter_by(email=email).first()
        if existing:
            raise HTTPException(status_code=409, detail="User already exists")
        is_admin = bool(payload.is_admin) if payload.is_admin is not None else False
        user = models.User(
            email=email,
            password_hash=hash_password(payload.password),
            active=True,
            is_admin=is_admin,
            restaurant_id=payload.restaurant_id,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return {
            "id": user.id,
            "email": user.email,
            "active": user.active,
            "is_admin": user.is_admin,
            "restaurant_id": user.restaurant_id,
        }
    finally:
        db.close()


@router.post("/users/assign")
def assign_user(payload: UserAssignPayload, _user=Depends(require_admin)):
    email = payload.email.strip().lower()
    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Invalid email")
    db = SessionLocal()
    try:
        user = db.query(models.User).filter_by(email=email).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user.is_admin:
            raise HTTPException(status_code=400, detail="Cannot assign admin user")
        user.restaurant_id = payload.restaurant_id
        db.commit()
        return {"ok": True, "restaurant_id": user.restaurant_id}
    finally:
        db.close()


@router.get("/users/by-restaurant/{restaurant_id}")
def users_by_restaurant(restaurant_id: int, _user=Depends(require_admin)):
    db = SessionLocal()
    try:
        rows = (
            db.query(models.User)
            .filter_by(restaurant_id=restaurant_id, is_admin=False)
            .order_by(models.User.id.desc())
            .all()
        )
        return [
            {
                "id": u.id,
                "email": u.email,
                "active": u.active,
                "restaurant_id": u.restaurant_id,
            }
            for u in rows
        ]
    finally:
        db.close()


@router.post("/users/{user_id}/status")
def set_user_status(user_id: int, payload: UserStatusPayload, _user=Depends(require_admin)):
    db = SessionLocal()
    try:
        user = db.get(models.User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user.is_admin:
            raise HTTPException(status_code=400, detail="Cannot change admin user")
        user.active = bool(payload.active)
        db.commit()
        if not user.active:
            revoke_user_tokens(user.id)
        return {"ok": True, "active": user.active}
    finally:
        db.close()


@router.post("/users/{user_id}/password")
def set_user_password(user_id: int, payload: UserPasswordPayload, _user=Depends(require_admin)):
    new_password = (payload.password or "").strip()
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="Password too short")
    db = SessionLocal()
    try:
        user = db.get(models.User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user.is_admin:
            raise HTTPException(status_code=400, detail="Cannot change admin user")
        user.password_hash = hash_password(new_password)
        db.commit()
        revoke_user_tokens(user.id)
        return {"ok": True}
    finally:
        db.close()


@router.put("/users/{user_id}")
def update_user(user_id: int, payload: UserUpdatePayload, _user=Depends(require_admin)):
    new_email = (payload.email or "").strip().lower()
    new_password = (payload.password or "").strip()
    if not new_email and not new_password:
        raise HTTPException(status_code=400, detail="Nothing to update")
    if new_email and ("@" not in new_email or "." not in new_email):
        raise HTTPException(status_code=400, detail="Invalid email")
    if new_password and len(new_password) < 6:
        raise HTTPException(status_code=400, detail="Password too short")
    db = SessionLocal()
    try:
        password_changed = False
        user = db.get(models.User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user.is_admin:
            raise HTTPException(status_code=400, detail="Cannot change admin user")
        if new_email and new_email != user.email:
            existing = db.query(models.User).filter_by(email=new_email).first()
            if existing and existing.id != user.id:
                raise HTTPException(status_code=409, detail="User already exists")
            user.email = new_email
        if new_password:
            user.password_hash = hash_password(new_password)
            password_changed = True
        db.commit()
        if password_changed:
            revoke_user_tokens(user.id)
        return {
            "ok": True,
            "user": {
                "id": user.id,
                "email": user.email,
                "active": user.active,
                "restaurant_id": user.restaurant_id,
            },
        }
    finally:
        db.close()


@router.delete("/users/{user_id}")
def delete_user(user_id: int, _user=Depends(require_admin)):
    db = SessionLocal()
    try:
        user = db.get(models.User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user.is_admin:
            raise HTTPException(status_code=400, detail="Cannot delete admin user")
        revoke_user_tokens(user.id)
        db.delete(user)
        db.commit()
        return {"ok": True}
    finally:
        db.close()
