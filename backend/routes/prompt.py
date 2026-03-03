import os

import httpx
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from database import SessionLocal
import models

router = APIRouter(prefix="/prompt")

class PromptUpdate(BaseModel):
    content: str


class PromptOptimizeRequest(BaseModel):
    prompt: str


PROMPT_OPTIMIZER_MODEL = "gpt-4o"


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


@router.post("/optimize-restaurant")
async def optimize_restaurant_prompt(payload: PromptOptimizeRequest):
    source_prompt = (payload.prompt or "").strip()
    if not source_prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")

    optimizer_rules = """You are a senior IVR prompt engineer specializing in restaurant voice systems.
Rewrite the provided restaurant prompt so it works reliably with a production voice backend.
Return only the final rewritten prompt text — no explanation, no markdown, no code fences.

CONTENT RULES — STRICT:
- Preserve ALL business facts exactly as given: restaurant name, address, area, nearby landmarks, opening hours, kosher certification, seating info, menu items, prices, extras, delivery zones, delivery fees, ordering rules, confirmation requirements, and any policies.
- NEVER invent, remove, or alter any menu item, ingredient, price, address, timing, delivery zone, or business rule.
- If the source prompt has a mandatory opening script — keep it word for word.
- If the source prompt has language switching behavior — keep it exactly.
- If the source prompt has fish/wrap/extra confirmation rules — keep them all.
- Every section of the source prompt must appear in the output. No section may be silently dropped.

TONE RULES:
- Caller-facing language must be warm, natural, and human — like a neighborhood receptionist.
- Never robotic. Never mention AI or system internals to the caller.
- Ask one question at a time.

COMPLETION FLOW RULES — NON-NEGOTIABLE:
- The prompt must include a clearly structured reservation/order completion flow.
- The model must first speak a warm goodbye confirmation OUT LOUD to the caller.
- After speaking goodbye, the model must output ONE silent backend marker line:
  ORDER_JSON: {"full_name":"...","date-arrival":"YYYY-MM-DD","time_arrival":"HH:MM","total_peoples":0,"contact_number":"..."}
- After ORDER_JSON, the model must output: <hangup>
- Explicitly state: ORDER_JSON and <hangup> are silent backend signals — NEVER spoken aloud, NEVER mentioned to caller.
- ORDER_JSON must be emitted ONLY ONCE at confirmed completion — never earlier, never repeated.
- No text of any kind after <hangup>."""


    runtime_context = """TARGET SYSTEM RUNTIME CONTEXT:

1) PROMPT COMPOSITION
- Restaurant prompt is saved as restaurant.ivr_text.
- Final system prompt = [global prompt] + blank line + [restaurant.ivr_text]
- Same composed prompt is used for both /chat and /voice-session endpoints.

2) RESERVATION CAPTURE CONTRACT — ALL 5 FIELDS REQUIRED
Backend saves the reservation only when ALL of these are present and valid:
  - full_name        → non-empty string, caller's name ONLY, no extra words
  - date-arrival     → format YYYY-MM-DD, non-empty
  - time_arrival     → format HH:MM (24-hour), non-empty
  - total_peoples    → integer greater than 0, numeric only
  - contact_number   → non-empty string

If any field is missing or invalid, the order will NOT be saved.
Prompt must enforce collection of all 5 fields before emitting ORDER_JSON.

3) MARKER PARSING CONTRACT
- Backend scans the full assistant text stream for the LAST occurrence of "ORDER_JSON:" marker.
- It extracts the first valid JSON object after that marker.
- Rules the prompt must enforce:
  a) Emit ORDER_JSON only at final confirmed completion — never mid-conversation
  b) Emit exactly ONE ORDER_JSON line total
  c) No extra text, punctuation, or characters on the ORDER_JSON line before or after the JSON
  d) JSON values must be clean — no labels, no units, no extra words inside values
  e) full_name value must contain ONLY the caller's name

4) HANGUP CONTRACT
- After ORDER_JSON, model must output: <hangup>
- Backend detects <hangup> tag to arm automatic call termination.
- Nothing must appear after <hangup> — no text, no punctuation, nothing.

5) FORMATTING CONSTRAINTS
  - date-arrival: YYYY-MM-DD only
  - time_arrival: HH:MM 24-hour only
  - total_peoples: integer only, no text
  - contact_number: digits only, no formatting
  - Do not round or modify any numeric values"""


    task_instructions = """YOUR TASK:

Rewrite the SOURCE_PROMPT below into a production-ready IVR system prompt using all rules and runtime context above.

CHECKLIST — your output MUST include all of the following:
✅ Mandatory opening script (word for word if provided in source)
✅ Restaurant name, full address, area, nearby landmarks
✅ All opening hours including closed days
✅ Kosher certification details
✅ Seating availability
✅ Complete menu with ALL items, ALL prices, ALL extras, ALL required selections
✅ Delivery zones, delivery fees, delivery options
✅ All ordering rules (fish confirmation, wrap confirmation, platter rules, etc.)
✅ Style rules (tone, one question at a time, no reading lists unless asked)
✅ Reservation/order collection flow with all 5 required fields
✅ Step-by-step confirmation with caller before finalizing
✅ Completion flow: spoken goodbye → silent ORDER_JSON → silent <hangup>
✅ Absolute rules section: never speak ORDER_JSON or <hangup>, never output before goodbye, nothing after <hangup>

OUTPUT FORMAT:
- Plain text only
- No markdown, no headers with #, no code fences
- Use clear section separators like === or ───
- Ready to paste directly as a system prompt"""

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": PROMPT_OPTIMIZER_MODEL,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": optimizer_rules},
                        {"role": "user", "content": runtime_context},
                        {"role": "user", "content": task_instructions},
                        {"role": "user", "content": f"SOURCE_PROMPT:\n{source_prompt}"},
                    ],
                },
            )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Optimization request failed: {exc}") from exc

    body = {}
    try:
        body = response.json()
    except ValueError:
        pass

    if response.status_code >= 400:
        detail = (
            (body.get("error") or {}).get("message")
            if isinstance(body, dict)
            else None
        ) or response.text or "Prompt optimization failed"
        raise HTTPException(status_code=502, detail=f"OpenAI: {detail}")

    choices = body.get("choices") if isinstance(body, dict) else None
    optimized = ""
    if isinstance(choices, list) and choices:
        optimized = str(((choices[0].get("message") or {}).get("content") or "")).strip()
    if optimized.startswith("```"):
        lines = optimized.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        optimized = "\n".join(lines).strip()

    if not optimized:
        raise HTTPException(status_code=502, detail="Prompt optimization returned empty content")

    return {"prompt": optimized}
