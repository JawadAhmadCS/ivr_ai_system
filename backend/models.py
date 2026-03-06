
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
    precall_notice_text = Column(Text)

class CallLog(Base):
    __tablename__ = "call_logs"
    id = Column(Integer, primary_key=True)
    restaurant_id = Column(Integer, index=True, nullable=True)
    restaurant = Column(String(200))
    caller = Column(String(50))
    call_sid = Column(String(100), index=True, nullable=True)
    duration = Column(Float, default=0)
    status = Column(String(50))
    created = Column(DateTime, default=datetime.utcnow)


class CallTranscript(Base):
    __tablename__ = "call_transcripts"
    id = Column(Integer, primary_key=True)
    call_sid = Column(String(100), index=True, nullable=False)
    restaurant_id = Column(Integer, index=True, nullable=True)
    restaurant = Column(String(200))
    caller = Column(String(50))
    speaker = Column(String(20), nullable=False)  # user | assistant
    content = Column(Text, nullable=False)
    seq = Column(Integer, default=0, nullable=False)
    created = Column(DateTime, default=datetime.utcnow)


class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    restaurant_id = Column(Integer, index=True, nullable=True)
    restaurant = Column(String(200))
    caller = Column(String(50))
    call_sid = Column(String(100))
    order_type = Column(String(50))
    full_name = Column(String(200))
    address = Column(String(300))
    house_number = Column(String(50))
    ordered_items = Column(Text)
    payment_method = Column(String(50))
    status = Column(String(50), default="pending_admin_review")
    raw_json = Column(Text)
    recording_sid = Column(String(100), nullable=True)
    recording_url = Column(String(500), nullable=True)
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
    restaurant_id = Column(Integer, index=True, nullable=True)
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


class ApiUsageLog(Base):
    __tablename__ = "api_usage_logs"
    id = Column(Integer, primary_key=True)
    restaurant_id = Column(Integer, index=True, nullable=True)
    endpoint = Column(String(80), nullable=False)
    model = Column(String(120), nullable=False)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cached_input_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    meta_json = Column(Text, nullable=True)
    created = Column(DateTime, default=datetime.utcnow)

