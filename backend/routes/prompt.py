from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from database import SessionLocal
import models

router = APIRouter(prefix="/prompt")

class PromptUpdate(BaseModel):
    content: str


def get_or_create_prompt(db):
    row = db.get(models.GlobalPrompt, 1)
    if row:
        return row
    row = models.GlobalPrompt(id=1, content="")
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.get("/")
def get_prompt():
    db = SessionLocal()
    try:
        row = db.get(models.GlobalPrompt, 1)
        return {"prompt": row.content if row else ""}
    finally:
        db.close()

@router.post("/save")
def save_prompt(
    content: str | None = None,
    payload: PromptUpdate | None = Body(default=None),
):
    text = content if content is not None else (payload.content if payload else None)
    if text is None:
        raise HTTPException(status_code=400, detail="content is required")
    db = SessionLocal()
    try:
        row = get_or_create_prompt(db)
        row.content = text
        db.commit()
        return {"saved": True}
    finally:
        db.close()
