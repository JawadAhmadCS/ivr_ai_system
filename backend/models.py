
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
