
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, Text
from datetime import datetime
from database import Base

class Restaurant(Base):
    __tablename__ = "restaurants"
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    phone = Column(String(50))
    active = Column(Boolean, default=True)
    ivr_text = Column(Text)

class CallLog(Base):
    __tablename__ = "call_logs"
    id = Column(Integer, primary_key=True)
    restaurant = Column(String(200))
    caller = Column(String(50))
    duration = Column(Float, default=0)
    status = Column(String(50))
    created = Column(DateTime, default=datetime.utcnow)


class GlobalPrompt(Base):
    __tablename__ = "global_prompts"
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created = Column(DateTime, default=datetime.utcnow)


class AuthToken(Base):
    __tablename__ = "auth_tokens"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    token = Column(String(255), unique=True, index=True, nullable=False)
    created = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    revoked = Column(Boolean, default=False)
