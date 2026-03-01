import os
import json
import asyncio
import logging
import websockets
from pathlib import Path
from datetime import datetime
from urllib.parse import quote_plus

import httpx
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect, Hangup

from database import engine, SessionLocal, ensure_schema
import models
from routes import restaurant, call_logs, dashboard, prompt, orders, auth as auth_routes
from auth import ensure_admin_user, require_auth

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


# ── Configuration ──────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PORT = int(os.getenv("PORT", 5050))
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
VOICE = os.getenv("VOICE", "shimmer")
LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").strip().upper()
REALTIME_LOG_EVERY_EVENT = _env_bool("REALTIME_LOG_EVERY_EVENT", False)

# VAD tuning
TURN_DETECTION_THRESHOLD = float(os.getenv("TURN_DETECTION_THRESHOLD", 0.4))
TURN_DETECTION_SILENCE_MS = int(os.getenv("TURN_DETECTION_SILENCE_MS", 400))
TURN_DETECTION_PREFIX_MS = int(os.getenv("TURN_DETECTION_PREFIX_MS", 100))
TURN_DETECTION_CREATE_RESPONSE = _env_bool("TURN_DETECTION_CREATE_RESPONSE", True)
TURN_DETECTION_INTERRUPT_RESPONSE = _env_bool("TURN_DETECTION_INTERRUPT_RESPONSE", True)

LLM_HANGUP_TOKEN = (os.getenv("LLM_HANGUP_TOKEN") or "<hangup>").strip()
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

ORDER_JSON_MARKER = "ORDER_JSON:"
ORDER_CAPTURE_RULES = (
    "Reservation capture requirements:\n"
    "- Collect full_name, date-arrival, time_arrival, total_peoples, and contact_number.\n"
    "- Do not complete capture until all required fields are present.\n"
    "- When complete, output one single-line JSON object prefixed with ORDER_JSON: using ONLY these keys:\n"
    '  full_name, date-arrival, time_arrival, total_peoples, contact_number.\n'
    "- Use valid JSON with double quotes. Do not read the JSON aloud to the caller."
)

BASE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = BASE_DIR / "prompts"
RESTAURANT_FILE = PROMPT_DIR / "restaurants.json"

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("ivr")
for _name in ("websockets", "websockets.client", "websockets.server", "urllib3", "httpx", "httpcore"):
    logging.getLogger(_name).setLevel(logging.INFO)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models.Base.metadata.create_all(bind=engine)
ensure_schema()

app.include_router(auth_routes.router)
app.include_router(restaurant.router, dependencies=[Depends(require_auth)])
app.include_router(call_logs.router, dependencies=[Depends(require_auth)])
app.include_router(orders.router, dependencies=[Depends(require_auth)])
app.include_router(dashboard.router, dependencies=[Depends(require_auth)])
app.include_router(prompt.router, dependencies=[Depends(require_auth)])

if not OPENAI_API_KEY:
    raise ValueError("Missing the OpenAI API key. Please set it in the .env file.")


# ── Prompt helpers ─────────────────────────────────────────────────────────────
def load_global_prompt() -> str:
    db = SessionLocal()
    try:
        row = db.get(models.GlobalPrompt, 1)
        return row.content if row and row.content else "You are a restaurant AI assistant."
    finally:
        db.close()


def load_restaurants():
    if RESTAURANT_FILE.exists():
        with open(RESTAURANT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def compose_system_prompt(restaurant_prompt: str | None) -> str:
    global_prompt = load_global_prompt()
    addon = (restaurant_prompt or "").strip()
    base = f"{global_prompt}\n{addon}" if addon else global_prompt

    hangup_rule = (
        "Only output `<hangup>` when the user clearly asks to end or disconnect the call. "
        "Never output `<hangup>` otherwise."
    )

    natural_tone = (
        "Tone and style rules:\n"
        "- Speak naturally and conversationally, like a warm human receptionist.\n"
        "- Keep responses short and to the point.\n"
        "- Never repeat what the caller just said back to them verbatim.\n"
        "- Do not use robotic filler phrases like 'Certainly!', 'Absolutely!', 'Of course!', 'Sure thing!'.\n"
        "- If the caller interrupts you, stop immediately and listen.\n"
        "- Never rush. Pause naturally between thoughts.\n"
        "- Ask only one question at a time."
    )

    return f"{base}\n\n{natural_tone}\n\n{hangup_rule}\n\n{ORDER_CAPTURE_RULES}"


def get_restaurant_prompt(restaurant_id: int | None) -> str:
    if not restaurant_id:
        return compose_system_prompt("")
    db = SessionLocal()
    try:
        r = db.get(models.Restaurant, restaurant_id)
        if not r or not r.active:
            return compose_system_prompt("")
        return compose_system_prompt(r.ivr_text or "")
    finally:
        db.close()


def get_restaurant_name(restaurant_id: int | None) -> str:
    if not restaurant_id:
        return "Unknown"
    db = SessionLocal()
    try:
        r = db.get(models.Restaurant, restaurant_id)
        return r.name if (r and r.name) else f"#{restaurant_id}"
    finally:
        db.close()


# ── DB helpers ─────────────────────────────────────────────────────────────────
def save_call_log(restaurant_id, restaurant_name, caller, duration, status):
    db = SessionLocal()
    try:
        db.add(models.CallLog(
            restaurant_id=restaurant_id,
            restaurant=restaurant_name,
            caller=caller,
            duration=duration,
            status=status,
        ))
        db.commit()
    finally:
        db.close()


def restaurant_exists_and_active(restaurant_id: int) -> bool:
    db = SessionLocal()
    try:
        r = db.get(models.Restaurant, restaurant_id)
        return bool(r and r.active)
    finally:
        db.close()


# ── OpenAI helpers ─────────────────────────────────────────────────────────────
def log_openai_status(context: str, working: bool, detail: str = ""):
    status = "WORKING" if working else "NOT WORKING"
    suffix = f" | {detail}" if detail else ""
    print(f"[OPENAI][{context}] {status}{suffix}")


def extract_openai_error(body, fallback_text: str = "") -> str:
    if isinstance(body, dict):
        error_obj = body.get("error")
        if isinstance(error_obj, dict):
            msg = error_obj.get("message")
            if msg:
                return str(msg)
    return fallback_text[:300] if fallback_text else "Unknown OpenAI error"


def check_openai_connectivity():
    url = "https://api.openai.com/v1/models/gpt-4o-mini"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        try:
            body = r.json()
        except ValueError:
            body = None
        if r.status_code >= 400:
            return False, f"HTTP {r.status_code}: {extract_openai_error(body, r.text)}"
        return True, "API reachable"
    except Exception as e:
        return False, str(e)


# ── Twilio helpers ─────────────────────────────────────────────────────────────
def complete_twilio_call(call_sid: str) -> bool:
    if not call_sid:
        return False
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        print("[TWILIO] Missing credentials; skipping REST hangup")
        return False
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.calls(call_sid).update(status="completed")
        print(f"[TWILIO] Call completed: {call_sid}")
        return True
    except Exception as e:
        print(f"[TWILIO] Failed to complete call {call_sid}: {e}")
        return False


# ── Text extraction helpers ────────────────────────────────────────────────────
def extract_text_candidates(response: dict) -> list[str]:
    if not isinstance(response, dict):
        return []
    event_type = str(response.get("type", ""))
    if event_type in {"response.output_audio.delta", "response.audio.delta"}:
        return []

    texts = []
    for key in ("delta", "text", "transcript"):
        value = response.get(key)
        if isinstance(value, str) and value:
            texts.append(value)

    for key in ("item", "response"):
        nested = response.get(key)
        if not isinstance(nested, dict):
            continue
        for nested_key in ("text", "transcript"):
            value = nested.get(nested_key)
            if isinstance(value, str) and value:
                texts.append(value)
        content = nested.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                for content_key in ("text", "transcript"):
                    value = part.get(content_key)
                    if isinstance(value, str) and value:
                        texts.append(value)
    return texts


def contains_hangup_token(response: dict, token: str) -> bool:
    if not token:
        return False
    marker = token.lower()
    for text in extract_text_candidates(response):
        if marker in text.lower():
            return True
    return False


# ── Order helpers ──────────────────────────────────────────────────────────────
def _clean_field(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "n/a", "na", "unknown"}:
        return ""
    return text


def _first_value(data: dict, keys: list[str]):
    for key in keys:
        if key in data:
            return data.get(key)
    return None


def normalize_order_payload(data: dict) -> tuple[dict, list[str]]:
    if not isinstance(data, dict):
        return {}, ["full_name", "date-arrival", "time_arrival", "total_peoples", "contact_number"]

    full_name        = _clean_field(_first_value(data, ["full_name", "fullName", "name", "customer_name", "customerName"]))
    date_arrival     = _clean_field(_first_value(data, ["date-arrival", "date_arrival", "arrival_date", "dateArrival"]))
    time_arrival     = _clean_field(_first_value(data, ["time_arrival", "arrival_time", "timeArrival", "time"]))
    contact_number   = _clean_field(_first_value(data, ["contact_number", "contactNumber", "phone", "phone_number"]))

    total_peoples_raw = _first_value(data, ["total_peoples", "total_people", "people", "party_size"])
    total_peoples = 0
    if isinstance(total_peoples_raw, int):
        total_peoples = total_peoples_raw
    else:
        total_text = _clean_field(total_peoples_raw)
        if total_text:
            digits = "".join(ch for ch in total_text if ch.isdigit())
            if digits:
                total_peoples = int(digits)

    missing = []
    if not full_name:       missing.append("full_name")
    if not date_arrival:    missing.append("date-arrival")
    if not time_arrival:    missing.append("time_arrival")
    if total_peoples <= 0:  missing.append("total_peoples")
    if not contact_number:  missing.append("contact_number")

    normalized = {
        "order_type":     "reservation",
        "full_name":      full_name,
        "date-arrival":   date_arrival,
        "time_arrival":   time_arrival,
        "total_peoples":  total_peoples,
        "contact_number": contact_number,
    }
    return normalized, missing


def parse_order_json_from_buffer(buffer: str) -> tuple[str, dict] | None:
    if not buffer:
        return None
    marker_idx = buffer.find(ORDER_JSON_MARKER)
    if marker_idx < 0:
        return None

    tail = buffer[marker_idx + len(ORDER_JSON_MARKER):].lstrip(" \n\t:")
    if tail.startswith("```"):
        fence_end = tail.find("\n")
        if fence_end != -1:
            tail = tail[fence_end + 1:]
    start = tail.find("{")
    if start < 0:
        return None

    depth = 0
    end = None
    for i in range(start, len(tail)):
        ch = tail[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None:
        return None

    json_str = tail[start: end + 1]
    try:
        data = json.loads(json_str)
    except ValueError:
        return None
    return json_str, data


def save_order(restaurant_id, restaurant_name, caller, call_sid, normalized, raw_json, status="completed"):
    db = SessionLocal()
    try:
        db.add(models.Order(
            restaurant_id=restaurant_id,
            restaurant=restaurant_name,
            caller=caller,
            call_sid=call_sid,
            order_type=normalized.get("order_type") or "reservation",
            full_name=normalized.get("full_name"),
            address=normalized.get("date-arrival"),
            house_number=normalized.get("time_arrival"),
            ordered_items=str(normalized.get("total_peoples") or ""),
            payment_method=normalized.get("contact_number"),
            status=status,
            raw_json=raw_json,
        ))
        db.commit()
    finally:
        db.close()


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_openai_check():
    ok, detail = check_openai_connectivity()
    log_openai_status("startup", ok, detail)
    ensure_admin_user()


@app.get("/health/openai", response_class=JSONResponse)
async def health_openai():
    ok, detail = check_openai_connectivity()
    log_openai_status("health", ok, detail)
    return {"working": ok, "detail": detail}


@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"status": "running"}


# ── Twilio webhook ─────────────────────────────────────────────────────────────
@app.api_route("/incoming-call", methods=["GET", "POST"])
@app.api_route("/incoming-call/{restaurant_id}", methods=["GET", "POST"])
async def incoming_call(request: Request, restaurant_id: int | None = None):
    response = VoiceResponse()

    if restaurant_id is None:
        q_id = request.query_params.get("restaurant_id")
        if q_id and q_id.isdigit():
            restaurant_id = int(q_id)

    params = dict(request.query_params)
    caller = str(params.get("From") or params.get("Caller") or "Unknown")
    call_sid = str(params.get("CallSid") or "")

    if restaurant_id is not None and not restaurant_exists_and_active(restaurant_id):
        response.say("Invalid restaurant id. Please contact support.", voice="alice")
        response.append(Hangup())
        return HTMLResponse(content=str(response), media_type="application/xml")

    host = request.headers.get("host")
    connect = Connect()
    qs = "edge=dublin"
    if caller:
        qs += f"&caller={quote_plus(caller)}"
    if call_sid:
        qs += f"&call_sid={quote_plus(call_sid)}"

    path = request.url.path or ""
    root_path = request.scope.get("root_path") or ""
    prefix = root_path
    if not prefix and path.startswith("/api/"):
        prefix = "/api"

    if restaurant_id is not None:
        connect.stream(url=f"wss://{host}{prefix}/media-stream/{restaurant_id}?{qs}")
    else:
        connect.stream(url=f"wss://{host}{prefix}/media-stream?{qs}")
    response.append(connect)

    return HTMLResponse(content=str(response), media_type="application/xml")


# ── Chat & session endpoints ───────────────────────────────────────────────────
@app.post("/chat")
async def chat_with_ai(data: dict, user=Depends(require_auth)):
    user_msg = data.get("message", "")
    restaurant_prompt = data.get("restaurant_prompt")
    restaurant_id = data.get("restaurant_id")
    if not user.is_admin:
        if not user.restaurant_id:
            raise HTTPException(status_code=403, detail="Forbidden")
        restaurant_id = user.restaurant_id

    final_prompt = compose_system_prompt(restaurant_prompt)
    if not (restaurant_prompt or "").strip() and restaurant_id is not None:
        try:
            final_prompt = get_restaurant_prompt(int(restaurant_id))
        except (TypeError, ValueError):
            final_prompt = compose_system_prompt("")

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": final_prompt},
            {"role": "user", "content": user_msg},
        ],
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    try:
        body = r.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="Invalid response from OpenAI API")

    if r.status_code >= 400:
        error_message = body.get("error", {}).get("message", "") if isinstance(body, dict) else r.text[:300]
        log_openai_status("chat", False, f"HTTP {r.status_code}: {error_message}")
        raise HTTPException(status_code=502, detail=f"OpenAI API error ({r.status_code}): {error_message}")

    choices = body.get("choices") if isinstance(body, dict) else None
    if not choices or not isinstance(choices, list):
        raise HTTPException(status_code=502, detail="OpenAI response missing choices")

    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    reply = message.get("content", "")
    if not isinstance(reply, str):
        reply = str(reply)

    log_openai_status("chat", True, "reply generated")
    return {"reply": reply}


@app.post("/orders/ingest")
async def ingest_order(data: dict, user=Depends(require_auth)):
    try:
        restaurant_id = data.get("restaurant_id")
        if not user.is_admin:
            if not user.restaurant_id:
                raise HTTPException(status_code=403, detail="Forbidden")
            restaurant_id = user.restaurant_id
        if restaurant_id is not None:
            try:
                restaurant_id = int(restaurant_id)
            except (TypeError, ValueError):
                restaurant_id = None

        normalized, missing = normalize_order_payload(data)
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing)}")

        restaurant_name = get_restaurant_name(restaurant_id)
        caller = str(data.get("caller") or "web")
        call_sid = str(data.get("call_sid") or "")
        raw_json = data.get("raw_json")
        if not isinstance(raw_json, str) or not raw_json.strip():
            raw_json = json.dumps({
                "full_name": normalized.get("full_name"),
                "date-arrival": normalized.get("date-arrival"),
                "time_arrival": normalized.get("time_arrival"),
                "total_peoples": normalized.get("total_peoples"),
                "contact_number": normalized.get("contact_number"),
            })

        save_order(restaurant_id, restaurant_name, caller, call_sid, normalized, raw_json, "completed")
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ORDER] ingest failed: {e}")
        raise HTTPException(status_code=500, detail="order ingest failed")


@app.api_route("/voice-session", methods=["GET", "POST"])
def create_voice_session(request: Request, data: dict | None = None, user=Depends(require_auth)):
    try:
        data = data or {}
        restaurant_prompt = data.get("restaurant_prompt")
        restaurant_id = data.get("restaurant_id") or request.query_params.get("restaurant_id")
        if not user.is_admin:
            if not user.restaurant_id:
                raise HTTPException(status_code=403, detail="Forbidden")
            restaurant_id = user.restaurant_id

        final_prompt = compose_system_prompt(restaurant_prompt)
        if not (restaurant_prompt or "").strip() and restaurant_id is not None:
            try:
                final_prompt = get_restaurant_prompt(int(restaurant_id))
            except (TypeError, ValueError):
                final_prompt = compose_system_prompt("")

        url = "https://api.openai.com/v1/realtime/sessions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": REALTIME_MODEL, "voice": VOICE, "instructions": final_prompt}

        r = requests.post(url, headers=headers, json=payload)
        try:
            body = r.json()
        except ValueError:
            raise HTTPException(status_code=502, detail="Invalid response from OpenAI realtime API")

        if r.status_code >= 400:
            error_message = extract_openai_error(body, r.text)
            raise HTTPException(status_code=502, detail=f"OpenAI realtime error ({r.status_code}): {error_message}")

        if not isinstance(body, dict) or not body.get("client_secret", {}).get("value"):
            raise HTTPException(status_code=502, detail="OpenAI realtime response missing client_secret")

        body["model"] = REALTIME_MODEL
        log_openai_status("voice-session", True, "session token issued")
        return body
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(e)
        raise HTTPException(status_code=500, detail="session error")


# ── Realtime session init ──────────────────────────────────────────────────────
async def init_session(openai_ws, instructions: str):
    """
    1. Send session.update with all config.
    2. Inject a hidden [call started] user message.
    3. Trigger response.create immediately.

    This means the AI greets the caller the instant the WebSocket
    is ready — no silence, no waiting for the user to speak first.
    """
    session_payload = {
        "type": "session.update",
        "session": {
            "model": REALTIME_MODEL,
            "instructions": instructions,
            "modalities": ["audio", "text"],
            "voice": VOICE,
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "input_audio_transcription": {"model": "whisper-1"},
            "turn_detection": {
                "type": "server_vad",
                "threshold": TURN_DETECTION_THRESHOLD,
                "silence_duration_ms": TURN_DETECTION_SILENCE_MS,
                "prefix_padding_ms": TURN_DETECTION_PREFIX_MS,
                "create_response": TURN_DETECTION_CREATE_RESPONSE,
                "interrupt_response": TURN_DETECTION_INTERRUPT_RESPONSE,
            },
        },
    }
    await openai_ws.send(json.dumps(session_payload))

    # Inject silent trigger so the model produces the opening greeting immediately
    await openai_ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "[call started]"}],
        },
    }))
    await openai_ws.send(json.dumps({"type": "response.create"}))


# ── Core WebSocket bridge ──────────────────────────────────────────────────────
async def handle_media_stream_with_id(websocket: WebSocket, restaurant_id: int | None):
    await websocket.accept()
    logger.info("[CALL] Twilio WS connected restaurant_id=%s", restaurant_id)

    instructions    = get_restaurant_prompt(restaurant_id)
    restaurant_name = get_restaurant_name(restaurant_id)
    caller          = websocket.query_params.get("caller") or "Unknown"
    call_sid        = websocket.query_params.get("call_sid") or ""

    try:
        async with websockets.connect(
            f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}&temperature={TEMPERATURE}",
            additional_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1",
            },
        ) as openai_ws:
            logger.info("[OPENAI] Realtime WS connected")
            await init_session(openai_ws, instructions)

            stream_sid          = None
            last_assistant_item = None
            response_active     = False
            response_cancelling = False
            start_ts            = None
            logged              = False
            call_already_ending = False
            order_buffer        = ""
            order_saved         = False
            marker_tail         = ""

            def log_call():
                nonlocal logged
                if logged:
                    return
                logged = True
                end_ts   = datetime.utcnow()
                duration = (end_ts - start_ts).total_seconds() if start_ts else 0
                status   = "missed" if duration < 3 else "completed"
                save_call_log(restaurant_id, restaurant_name, caller, float(duration), status)

            # ── Safe send wrappers (never crash the loop) ──────────────────────
            async def safe_openai_send(payload: dict):
                try:
                    await openai_ws.send(json.dumps(payload))
                except Exception as e:
                    logger.debug("[OPENAI] send ignored: %s", e)

            async def safe_twilio_send(payload: dict):
                try:
                    await websocket.send_json(payload)
                except Exception as e:
                    logger.debug("[TWILIO] send ignored: %s", e)

            # ── Interrupt: stop AI speaking the instant user starts talking ─────
            async def interrupt_assistant():
                nonlocal response_active, response_cancelling, last_assistant_item
                if response_cancelling:
                    return
                response_cancelling = True
                response_active     = False
                last_assistant_item = None
                logger.info("[CALL] Interrupting assistant")
                # 1. Cancel OpenAI generation
                await safe_openai_send({"type": "response.cancel"})
                # 2. Flush any audio already buffered at Twilio
                if stream_sid:
                    await safe_twilio_send({"event": "clear", "streamSid": stream_sid})

            # ── Receive audio from Twilio ──────────────────────────────────────
            async def receive_twilio():
                nonlocal stream_sid, start_ts, call_sid

                try:
                    async for message in websocket.iter_text():
                        data  = json.loads(message)
                        event = data.get("event")

                        if event == "start":
                            stream_sid = data["start"]["streamSid"]
                            start_ts   = datetime.utcnow()
                            start_obj  = data.get("start", {})
                            if not call_sid:
                                call_sid = str(start_obj.get("callSid") or "")
                            logger.info("[CALL] Stream started sid=%s", stream_sid)

                        elif event == "media":
                            await safe_openai_send({
                                "type":  "input_audio_buffer.append",
                                "audio": data["media"]["payload"],
                            })

                        elif event == "stop":
                            logger.info("[CALL] Stream stop sid=%s", stream_sid)
                            try:
                                await openai_ws.close()
                            except Exception:
                                pass
                            break

                except WebSocketDisconnect:
                    logger.info("[CALL] Twilio WS disconnected sid=%s", stream_sid)
                    try:
                        await openai_ws.close()
                    except Exception:
                        pass
                except RuntimeError as e:
                    if "WebSocket is not connected" not in str(e):
                        raise
                    logger.info("[CALL] Twilio WS already closed sid=%s", stream_sid)
                    try:
                        await openai_ws.close()
                    except Exception:
                        pass

            # ── End call ──────────────────────────────────────────────────────
            async def end_call_from_assistant():
                nonlocal call_already_ending
                if call_already_ending:
                    return
                call_already_ending = True
                logger.info("[CALL] Hangup token — ending call")
                if call_sid:
                    await asyncio.to_thread(complete_twilio_call, call_sid)
                try:
                    await websocket.close()
                except Exception:
                    pass
                try:
                    await openai_ws.close()
                except Exception:
                    pass

            # ── Receive from OpenAI → forward audio to Twilio ──────────────────
            async def send_twilio():
                nonlocal last_assistant_item, response_active, response_cancelling
                nonlocal order_buffer, order_saved, marker_tail

                async for raw_msg in openai_ws:
                    response   = json.loads(raw_msg)
                    event_type = response.get("type")

                    if REALTIME_LOG_EVERY_EVENT and event_type:
                        logger.info("[OPENAI] event=%s", event_type)

                    # ── Response lifecycle ─────────────────────────────────────
                    if event_type == "response.created":
                        response_active     = True
                        response_cancelling = False

                    elif event_type in {"response.done", "response.cancelled"}:
                        response_active     = False
                        response_cancelling = False
                        last_assistant_item = None

                    # ── Error events ───────────────────────────────────────────
                    elif event_type == "error":
                        err      = response.get("error") if isinstance(response.get("error"), dict) else {}
                        code     = err.get("code", "unknown")
                        msg_text = err.get("message", "unknown error")
                        # Benign cancel-race errors — just reset flag and continue
                        if code in ("response_cancel_not_active", "response_already_cancelled", "item_truncated"):
                            response_cancelling = False
                            continue
                        logger.error("[OPENAI] error %s: %s", code, msg_text)
                        continue

                    # ── INTERRUPTION: user starts speaking ─────────────────────
                    # We catch this at the EARLIEST possible event so latency is
                    # minimised — audio stops within one network round trip.
                    if event_type == "input_audio_buffer.speech_started":
                        logger.info("[CALL] speech_started — interrupting")
                        await interrupt_assistant()

                    # ── FALLBACK: ensure response after user finishes speaking ──
                    # create_response=True handles this automatically via VAD,
                    # but we keep a manual fallback for reliability.
                    elif event_type == "input_audio_buffer.speech_stopped":
                        if not response_active and not response_cancelling:
                            logger.debug("[OPENAI] speech_stopped fallback → response.create")
                            await safe_openai_send({"type": "response.create"})
                            response_active = True

                    # ── Order JSON capture ─────────────────────────────────────
                    if not order_saved:
                        for text in extract_text_candidates(response):
                            if order_buffer:
                                order_buffer += text
                            else:
                                combined = marker_tail + text
                                if ORDER_JSON_MARKER not in combined:
                                    marker_tail = combined[-(len(ORDER_JSON_MARKER) - 1):]
                                    continue
                                idx          = combined.find(ORDER_JSON_MARKER)
                                order_buffer = combined[idx:]
                            if len(order_buffer) > 10_000:
                                order_buffer = order_buffer[-10_000:]
                            parsed = parse_order_json_from_buffer(order_buffer)
                            if not parsed:
                                continue
                            raw_json, raw_data = parsed
                            normalized, missing = normalize_order_payload(raw_data)
                            if missing:
                                logger.warning("[ORDER] Incomplete JSON missing=%s", missing)
                                order_buffer = ""
                                marker_tail  = ""
                                continue
                            try:
                                save_order(
                                    restaurant_id=restaurant_id,
                                    restaurant_name=restaurant_name,
                                    caller=caller,
                                    call_sid=call_sid,
                                    normalized=normalized,
                                    raw_json=raw_json,
                                    status="completed",
                                )
                                order_saved = True
                                logger.info("[ORDER] Saved restaurant=%s call_sid=%s",
                                            restaurant_name, call_sid or "n/a")
                            except Exception as e:
                                logger.exception("[ORDER] Save failed: %s", e)
                            break

                    # ── Hangup token ───────────────────────────────────────────
                    if contains_hangup_token(response, LLM_HANGUP_TOKEN):
                        if response_active:
                            await safe_openai_send({"type": "response.cancel"})
                            response_active = False
                        if stream_sid:
                            await safe_twilio_send({"event": "clear", "streamSid": stream_sid})
                        await end_call_from_assistant()
                        break

                    # ── Stream audio → Twilio ──────────────────────────────────
                    if event_type in {"response.output_audio.delta", "response.audio.delta"}:
                        audio = response.get("delta")
                        if isinstance(audio, str) and stream_sid:
                            await safe_twilio_send({
                                "event":    "media",
                                "streamSid": stream_sid,
                                "media":    {"payload": audio},
                            })
                        if response.get("item_id"):
                            last_assistant_item = response["item_id"]
                            response_active     = True

            # ── Run both coroutines concurrently ───────────────────────────────
            try:
                await asyncio.gather(receive_twilio(), send_twilio())
            finally:
                log_call()

    except Exception:
        logger.exception("[CALL] Bridge failed restaurant_id=%s caller=%s call_sid=%s",
                         restaurant_id, caller, call_sid)
        try:
            await websocket.close()
        except Exception:
            pass


# ── WebSocket routes ───────────────────────────────────────────────────────────
@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    q_id = websocket.query_params.get("restaurant_id")
    restaurant_id = int(q_id) if q_id and q_id.isdigit() else None
    await handle_media_stream_with_id(websocket, restaurant_id)


@app.websocket("/media-stream/{restaurant_id}")
async def handle_media_stream_restaurant(websocket: WebSocket, restaurant_id: int):
    await handle_media_stream_with_id(websocket, restaurant_id)


@app.get("/media-stream")
@app.get("/media-stream/{restaurant_id}")
def media_stream_http_guard(restaurant_id: int | None = None):
    return JSONResponse(
        status_code=426,
        content={
            "error":  "WebSocket required",
            "detail": "Use the Twilio Voice webhook /incoming-call to initiate a WebSocket media stream.",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)