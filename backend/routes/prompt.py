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

═══════════════════════════════════════════════════════
CONTENT RULES — ABSOLUTE AND NON-NEGOTIABLE
═══════════════════════════════════════════════════════

You MUST preserve and explicitly include EVERY piece of information from the source prompt.
Missing even one item is a failure. Go through the source line by line before writing.

Business facts to carry over verbatim:
- Restaurant name (exact spelling and casing)
- Full address, area name, nearby landmarks (all of them)
- Opening hours for EVERY day listed, including closed days
- Kosher certification body and scope
- Seating / dine-in availability
- Every menu item name, price, and category (starters, rolls, platters, drinks)
- Every built-in ingredient and whether it is removable
- Every required selection (fish type, filling type, salmon type, etc.) with all options
- Every optional upgrade with its price (+5₪, +10₪, +15₪, etc.)
- Every sauce rule (max count per item)
- Every platter configuration (how many rolls, fixed price, never price individually)
- Every drink item with its price, type (can / bottle / other), and flavor
- Delivery zones (city or area restrictions)
- Delivery fee options (building entrance vs to door) with exact amounts
- All delivery types: building / door / pickup / dine_in
- Payment methods accepted and validation rules
- All mandatory confirmation scripts (fish confirmation wording, wrap confirmation wording)
- All call handling scripts (noise, reconnect, silence)
- Language behavior (default language, when to switch)
- Mandatory opening script — carry it over WORD FOR WORD, no paraphrasing

NEVER invent any item, ingredient, price, address, landmark, hour, zone, fee, or rule.
NEVER remove any menu item, even if it seems simple or duplicate.
NEVER merge or abbreviate sections that were separate in the source.
NEVER silently drop any section. Every section in the source must appear in the output.

═══════════════════════════════════════════════════════
TONE RULES
═══════════════════════════════════════════════════════

- Caller-facing language must be warm, natural, and human — like a neighborhood receptionist.
- Never robotic. Never clinical. Never list-like when speaking to caller.
- Never mention AI, system, bot, or any technical internals to the caller.
- Ask one question at a time.
- Short responses. Guide the call confidently.

═══════════════════════════════════════════════════════
STRUCTURE RULES
═══════════════════════════════════════════════════════

- Use clear section separators (=== or ───) between every section.
- Plain text only — no markdown headers (#), no code fences, no bullet symbols.
- Output must be ready to paste directly as a system prompt with zero editing.

═══════════════════════════════════════════════════════
COMPLETION FLOW RULES — NON-NEGOTIABLE
═══════════════════════════════════════════════════════

The model must handle two paths: reservation and food order.

For BOTH paths, the completion sequence is strictly:

  STEP 1: Confirm all collected details with the caller out loud.
          Wait for explicit YES before proceeding.
          If caller corrects anything — update and re-confirm.

  STEP 2: Speak a warm goodbye confirmation OUT LOUD to the caller.
          (e.g. "Perfect! Your reservation is confirmed. We look forward to welcoming you...")

  STEP 3: Output ONE silent backend marker line — never spoken, never mentioned:
          ORDER_JSON: { ... valid JSON with all required fields ... }

  STEP 4: Output the hangup tag — never spoken, never mentioned:
          <hangup>

  STEP 5: Absolutely nothing after <hangup>. Not a space. Not a newline. Nothing.

ORDER_JSON rules:
- Emit ONLY at confirmed completion — never mid-conversation, never as a preview.
- Emit exactly ONCE total across the entire conversation.
- No extra text, punctuation, or characters on the ORDER_JSON line before or after the JSON.
- JSON values must be clean — no labels, no units, no extra words inside values.
- full_name must contain ONLY the caller's name — no titles, no extra words.
- date-arrival: YYYY-MM-DD format only.
- time_arrival: HH:MM 24-hour format only.
- total_peoples: integer only — no text, no units.
- contact_number: digits only — no spaces, no dashes, no formatting.
- card_no: exactly 16 digits after removing spaces/hyphens, or empty string "" for cash.
- delivery_type values: building / door / pickup / dine_in only.
- order_type values: reservation / pickup / delivery / dine_in only.
- payment_method values: card / cash only.
- For pickup or dine_in: delivery_address fields must be null.

ALL 5 reservation fields are required before emitting ORDER_JSON:
  full_name, date-arrival, time_arrival, total_peoples, contact_number.
If any field is missing or invalid, the reservation WILL NOT be saved — collect all fields first.

═══════════════════════════════════════════════════════
ABSOLUTE RULES — NEVER VIOLATE UNDER ANY CIRCUMSTANCE
═══════════════════════════════════════════════════════

- NEVER speak ORDER_JSON aloud or mention it to the caller.
- NEVER speak <hangup> aloud or mention it to the caller.
- NEVER output ORDER_JSON before the spoken goodbye has been delivered.
- NEVER output ORDER_JSON more than once.
- NEVER output anything after <hangup>.
- NEVER finalize without explicit caller confirmation.
- NEVER skip the mandatory opening script.
- NEVER mention you are a system, bot, or AI.
- NEVER price platter rolls individually — platter price is always fixed.
- NEVER read full card number aloud — confirm only the last 4 digits.
- NEVER add menu items, ingredients, or prices not present in the source prompt.
- NEVER round or modify any numeric value."""

    runtime_context = """
═══════════════════════════════════════════════════════
TARGET SYSTEM RUNTIME CONTEXT — READ BEFORE WRITING
═══════════════════════════════════════════════════════

This context describes exactly how the backend processes the prompt output.
Every rule here maps to a real system behavior. Violating any rule = silent data loss or failed order.

───────────────────────────────────────────────────────
1. PROMPT COMPOSITION
───────────────────────────────────────────────────────

- Restaurant prompt is saved as: restaurant.ivr_text
- Final system prompt = [global prompt] + blank line + [restaurant.ivr_text]
- The same composed prompt is used for BOTH /chat and /voice-session endpoints.
- The rewritten prompt must work correctly as the restaurant.ivr_text portion alone.
- Do not assume any global context will fill in missing info — everything must be self-contained.

───────────────────────────────────────────────────────
2. RESERVATION CAPTURE CONTRACT — ALL 5 FIELDS MANDATORY
───────────────────────────────────────────────────────

The backend saves a reservation ONLY when ALL 5 fields are present and valid.
If even one field is missing, malformed, or empty — the reservation is silently dropped.

Required fields and their exact format:

  full_name        → non-empty string — caller's name ONLY — no titles, no extra words
  date-arrival     → YYYY-MM-DD format — non-empty — no slashes, no dots, no natural language
  time_arrival     → HH:MM 24-hour format — non-empty — no AM/PM, no "half past"
  total_peoples    → integer greater than 0 — digits only — no text, no units
  contact_number   → non-empty string — digits only — no spaces, dashes, or formatting

The prompt MUST enforce collection of all 5 fields before emitting ORDER_JSON.
The prompt MUST validate format before emitting — ask again if caller gives invalid input.

───────────────────────────────────────────────────────
3. ORDER CAPTURE CONTRACT — FOOD ORDERS
───────────────────────────────────────────────────────

Food order JSON must include:

  order_type          → reservation / pickup / delivery / dine_in
  full_name           → caller's name only — no extra words
  contact_number      → digits only
  delivery_type       → building / door / pickup / dine_in
  delivery_address    → object with city, street, house_number
                        → set all three to null for pickup or dine_in
  ordered_items       → array of objects: [{"item_name":"...","quantity":"..."}]
  payment_method      → card / cash — no other values accepted
  card_no             → exactly 16 digits after removing spaces/hyphens
                        → empty string "" if payment_method is cash

───────────────────────────────────────────────────────
4. MARKER PARSING CONTRACT — HOW BACKEND READS ORDER_JSON
───────────────────────────────────────────────────────

The backend scans the FULL assistant text stream and finds the LAST occurrence of "ORDER_JSON:".
It then extracts the first valid JSON object that follows that marker.

Rules the prompt MUST enforce:

  a) Emit ORDER_JSON ONLY at final confirmed completion — NEVER mid-conversation.
  b) Emit EXACTLY ONE ORDER_JSON line total — no repeats, no previews, no partials.
  c) The ORDER_JSON line must contain NOTHING before "ORDER_JSON:" and NOTHING after the closing "}"
     — no punctuation, no newline text, no trailing characters.
  d) All JSON values must be clean — no labels, no units, no extra words embedded in values.
     WRONG:  "total_peoples": "3 people"
     RIGHT:  "total_peoples": 3
     WRONG:  "time_arrival": "8:30 PM"
     RIGHT:  "time_arrival": "20:30"
  e) full_name value must contain ONLY the caller's name.
     WRONG:  "full_name": "Name: David Cohen"
     RIGHT:  "full_name": "David Cohen"

───────────────────────────────────────────────────────
5. HANGUP CONTRACT
───────────────────────────────────────────────────────

- Immediately after ORDER_JSON, output exactly: <hangup>
- Backend detects <hangup> to arm automatic call termination.
- <hangup> must appear on its own line immediately after ORDER_JSON.
- NOTHING may appear after <hangup> — no text, no space, no newline, no punctuation.
- <hangup> is a silent backend signal — NEVER spoken aloud, NEVER mentioned to caller.

───────────────────────────────────────────────────────
6. FORMATTING CONSTRAINTS — EXACT FORMATS REQUIRED
───────────────────────────────────────────────────────

  date-arrival       → YYYY-MM-DD only         (e.g. 2025-08-14)
  time_arrival       → HH:MM 24-hour only       (e.g. 20:30)
  total_peoples      → integer only             (e.g. 4)
  contact_number     → digits only              (e.g. 0501234567)
  card_no            → 16 digits only           (e.g. 4111111111111111)
                       or empty string ""       (if cash)
  delivery_type      → building / door / pickup / dine_in
  order_type         → reservation / pickup / delivery / dine_in
  payment_method     → card / cash

Do NOT round or modify any numeric value.
Do NOT convert natural language time or date — ask caller to confirm if format is unclear.
"""


    task_instructions = """
═══════════════════════════════════════════════════════
YOUR TASK — READ FULLY BEFORE WRITING A SINGLE WORD
═══════════════════════════════════════════════════════

Rewrite the SOURCE_PROMPT below into a production-ready IVR system prompt.
Apply all rules in optimizer_rules and all contracts in runtime_context.

───────────────────────────────────────────────────────
BEFORE YOU WRITE — MANDATORY PRE-CHECK
───────────────────────────────────────────────────────

Read the source prompt completely first. Then verify:

  □ Did I find the mandatory opening script?        → Will copy word for word.
  □ Did I find the restaurant name and address?     → Will include full detail.
  □ Did I find ALL opening hours including closed?  → Will include every day.
  □ Did I find kosher certification?                → Will include authority and scope.
  □ Did I find seating info?                        → Will include.
  □ Did I find EVERY menu item?                     → Will list every one with price.
  □ Did I find ALL extras, upgrades, and prices?    → Will include each with exact amount.
  □ Did I find ALL required selections per item?    → Will mark each as mandatory.
  □ Did I find platter configurations?              → Will include rolls count and fixed price.
  □ Did I find ALL drinks with prices?              → Will include every flavor and type.
  □ Did I find delivery zones and fee options?      → Will include building vs door amounts.
  □ Did I find all confirmation scripts?            → Will include fish confirm, wrap confirm.
  □ Did I find all call handling scripts?           → Will include noise, reconnect, silence.
  □ Did I find language behavior rules?             → Will preserve exactly.
  □ Did I find all ordering and POS rules?          → Will preserve every rule.

If any item above is missing from the source — note it explicitly, do NOT invent it.

───────────────────────────────────────────────────────
COMPLETION FLOW — WRITE THIS EXACTLY AS SHOWN
───────────────────────────────────────────────────────

The output prompt MUST instruct the model to follow this exact sequence at end of every call:

  1. Confirm all details with caller out loud — wait for explicit YES.
     If caller corrects anything — update and re-confirm before continuing.

  2. Speak goodbye confirmation out loud.
     Reservation example:
       "Perfect! Your table is reserved. We look forward to welcoming you at [Restaurant]. Thank you, goodbye!"
     Order example:
       "Great! Your order has been received. Thank you for calling [Restaurant]. Enjoy your meal!"

  3. Immediately and silently output:
     ORDER_JSON: {"field":"value", ...}

  4. Immediately and silently output:
     <hangup>

  5. Output NOTHING after <hangup>. Not a word, not a space.

The output prompt MUST explicitly state:
  - ORDER_JSON is a silent backend signal — NEVER speak it aloud, NEVER reference it to caller.
  - <hangup> is a silent backend signal — NEVER speak it aloud, NEVER reference it to caller.
  - ORDER_JSON must be emitted ONLY ONCE — never mid-conversation, never as a preview.
  - Nothing may appear after <hangup>.

───────────────────────────────────────────────────────
OUTPUT FORMAT REQUIREMENTS
───────────────────────────────────────────────────────

  - Plain text only.
  - No markdown, no # headers, no code fences, no bullet symbols.
  - Use === or ─── as section separators.
  - Every section from the source must appear as a named section in the output.
  - Ready to paste directly as a system prompt — zero editing required.

───────────────────────────────────────────────────────
FINAL SELF-CHECK — RUN BEFORE RETURNING OUTPUT
───────────────────────────────────────────────────────

Before returning the rewritten prompt, verify:

  ✅ Mandatory opening script included — word for word.
  ✅ Restaurant name, full address, area, all nearby landmarks.
  ✅ All opening hours including every closed day explicitly stated.
  ✅ Kosher certification — authority name and scope.
  ✅ Seating availability.
  ✅ Complete menu — every item, every price, every extra, every required selection.
  ✅ Every sauce rule, wrap rule, platter rule, fish confirmation rule.
  ✅ All drink items — every flavor, every price, every type (can/bottle/other).
  ✅ Delivery zones, both delivery fee options with exact amounts.
  ✅ All four delivery types: building / door / pickup / dine_in.
  ✅ Payment methods and card validation rules.
  ✅ All mandatory confirmation scripts (fish, wrap, card).
  ✅ All call handling scripts (noise, reconnect, silence).
  ✅ Language behavior preserved exactly.
  ✅ PATH A (reservation) — all 5 fields, confirm step, spoken goodbye, ORDER_JSON, <hangup>.
  ✅ PATH B (food order) — all steps 1–8, spoken goodbye, ORDER_JSON, <hangup>.
  ✅ Absolute rules section — ORDER_JSON silent, <hangup> silent, nothing after hangup.
  ✅ JSON format section — all field formats with examples.
  ✅ No section from source was dropped or merged silently.
  ✅ No invented items, prices, ingredients, or rules.

If any checkbox above fails — fix it before returning.
"""

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
