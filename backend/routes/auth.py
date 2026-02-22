from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from database import SessionLocal
import models
from auth import verify_password, create_token, ensure_admin_user, require_auth, require_token

router = APIRouter(prefix="/auth")


class LoginPayload(BaseModel):
    email: str
    password: str


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
    return {"email": user.email, "id": user.id}
