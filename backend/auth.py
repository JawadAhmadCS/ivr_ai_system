import os
import secrets
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext

from database import SessionLocal
import models

# Use PBKDF2 to avoid bcrypt backend issues on some Windows setups.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)

TOKEN_TTL_HOURS = int(os.getenv("TOKEN_TTL_HOURS", "24"))
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def ensure_admin_user() -> None:
    if not ADMIN_EMAIL or not ADMIN_PASSWORD:
        return
    db = SessionLocal()
    try:
        user = db.query(models.User).filter_by(email=ADMIN_EMAIL).first()
        if not user:
            user = models.User(
                email=ADMIN_EMAIL.strip().lower(),
                password_hash=hash_password(ADMIN_PASSWORD),
                active=True,
                is_admin=True,
            )
            db.add(user)
            db.commit()
    finally:
        db.close()


def create_token(user_id: int) -> models.AuthToken:
    token_value = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=TOKEN_TTL_HOURS)
    db = SessionLocal()
    try:
        token = models.AuthToken(
            user_id=user_id,
            token=token_value,
            expires_at=expires_at,
        )
        db.add(token)
        db.commit()
        db.refresh(token)
        return token
    finally:
        db.close()


def revoke_token(token_value: str) -> None:
    db = SessionLocal()
    try:
        token = db.query(models.AuthToken).filter_by(token=token_value).first()
        if token:
            token.revoked = True
            db.commit()
    finally:
        db.close()


def _get_token_record(token_value: str) -> tuple[models.User, models.AuthToken] | None:
    db = SessionLocal()
    try:
        token = (
            db.query(models.AuthToken)
            .filter_by(token=token_value, revoked=False)
            .first()
        )
        if not token:
            return None
        if token.expires_at and token.expires_at <= datetime.utcnow():
            token.revoked = True
            db.commit()
            return None
        user = db.get(models.User, token.user_id)
        if not user or not user.active:
            return None
        return user, token
    finally:
        db.close()


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> models.User:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Unauthorized")
    token_value = credentials.credentials
    record = _get_token_record(token_value)
    if not record:
        raise HTTPException(status_code=401, detail="Unauthorized")
    user, _token = record
    return user


def require_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Unauthorized")
    token_value = credentials.credentials
    record = _get_token_record(token_value)
    if not record:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return token_value
