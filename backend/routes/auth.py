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


@router.post("/logout")
def logout(token_value: str = Depends(require_token)):
    from auth import revoke_token

    revoke_token(token_value)
    return {"ok": True}


@router.get("/me")
def me(user=Depends(require_auth)):
    return {
        "email": user.email,
        "id": user.id,
        "is_admin": user.is_admin,
        "restaurant_id": user.restaurant_id,
    }


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
