from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from pathlib import Path

router = APIRouter(prefix="/prompt")

BASE_DIR = Path(__file__).resolve().parents[1]
PROMPT_DIR = BASE_DIR / "prompts"
GLOBAL_FILE = PROMPT_DIR / "global_prompt.txt"

class PromptUpdate(BaseModel):
    content: str

@router.get("/")
def get_prompt():
    if GLOBAL_FILE.exists():
        return {"prompt": GLOBAL_FILE.read_text(encoding="utf-8")}
    return {"prompt": ""}

@router.post("/save")
def save_prompt(
    content: str | None = None,
    payload: PromptUpdate | None = Body(default=None),
):
    text = content if content is not None else (payload.content if payload else None)
    if text is None:
        raise HTTPException(status_code=400, detail="content is required")
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_FILE.write_text(text, encoding="utf-8")
    return {"saved": True}
