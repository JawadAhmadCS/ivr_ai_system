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
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
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
OPENAI_API_KEY           = os.getenv("OPENAI_API_KEY")
PORT                     = int(os.getenv("PORT", 5050))
REALTIME_MODEL           = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")
TEMPERATURE              = float(os.getenv("TEMPERATURE", 0.6))
VOICE                    = os.getenv("VOICE", "alloy")
LOG_LEVEL                = (os.getenv("LOG_LEVEL") or "INFO").strip().upper()
REALTIME_LOG_EVERY_EVENT = _env_bool("REALTIME_LOG_EVERY_EVENT", False)

TURN_DETECTION_THRESHOLD      = float(os.getenv("TURN_DETECTION_THRESHOLD", 0.7))
TURN_DETECTION_SILENCE_MS     = int(os.getenv("TURN_DETECTION_SILENCE_MS", 600))
TURN_DETECTION_PREFIX_MS      = int(os.getenv("TURN_DETECTION_PREFIX_MS", 300))
TURN_DETECTION_CREATE_RESPONSE    = _env_bool("TURN_DETECTION_CREATE_RESPONSE", True)
TURN_DETECTION_INTERRUPT_RESPONSE = _env_bool("TURN_DETECTION_INTERRUPT_RESPONSE", True)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN")

ORDER_JSON_MARKER = "ORDER_JSON:"

BASE_DIR        = Path(__file__).resolve().parent
PROMPT_DIR      = BASE_DIR / "prompts"
RESTAURANT_FILE = PROMPT_DIR / "restaurants.json"
FRONTEND_DIR    = BASE_DIR.parent / "frontend"
DASHBOARD_FILE  = FRONTEND_DIR / "dashboard.html"

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("ivr")
for _noisy in ("websockets", "websockets.client", "websockets.server",
               "urllib3", "httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

models.Base.metadata.create_all(bind=engine)
ensure_schema()

app.include_router(auth_routes.router)
app.include_router(restaurant.router,  dependencies=[Depends(require_auth)])
app.include_router(call_logs.router,   dependencies=[Depends(require_auth)])
app.include_router(orders.router,      dependencies=[Depends(require_auth)])
app.include_router(dashboard.router,   dependencies=[Depends(require_auth)])
app.include_router(prompt.router,      dependencies=[Depends(require_auth)])
app.include_router(auth_routes.router, prefix="/api")
app.include_router(restaurant.router,  prefix="/api", dependencies=[Depends(require_auth)])
app.include_router(call_logs.router,   prefix="/api", dependencies=[Depends(require_auth)])
app.include_router(orders.router,      prefix="/api", dependencies=[Depends(require_auth)])
app.include_router(dashboard.router,   prefix="/api", dependencies=[Depends(require_auth)])
app.include_router(prompt.router,      prefix="/api", dependencies=[Depends(require_auth)])

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")


# ── Prompt ─────────────────────────────────────────────────────────────────────
def load_global_prompt() -> str:
    db = SessionLocal()
    try:
        row = db.get(models.GlobalPrompt, 1)
        return (row.content or "").strip() if row else ""
    finally:
        db.close()


def compose_system_prompt(restaurant_prompt: str | None) -> str:
    global_prompt = load_global_prompt() or "You are a friendly restaurant reservation assistant."
    addon = (restaurant_prompt or "").strip()
    base  = f"{global_prompt}\n\n{addon}" if addon else global_prompt

    reservation_instructions = """\

"""

    return f"{base}\n{reservation_instructions}"


def get_restaurant_prompt(restaurant_id: int | None) -> str:
    if not restaurant_id:
        return compose_system_prompt("")
    db = SessionLocal()
    try:
        r = db.get(models.Restaurant, restaurant_id)
        return compose_system_prompt(r.ivr_text or "" if r and r.active else "")
    finally:
        db.close()


def get_restaurant_name(restaurant_id: int | None) -> str:
    if not restaurant_id:
        return "Unknown"
    db = SessionLocal()
    try:
        r = db.get(models.Restaurant, restaurant_id)
        return (r.name or f"#{restaurant_id}") if r else f"#{restaurant_id}"
    finally:
        db.close()


# ── DB helpers ─────────────────────────────────────────────────────────────────
def save_call_log(restaurant_id, restaurant_name, caller, duration, status):
    db = SessionLocal()
    try:
        db.add(models.CallLog(
            restaurant_id=restaurant_id, restaurant=restaurant_name,
            caller=caller, duration=duration, status=status,
        ))
        db.commit()
    finally:
        db.close()


def restaurant_exists_and_active(rid: int) -> bool:
    db = SessionLocal()
    try:
        r = db.get(models.Restaurant, rid)
        return bool(r and r.active)
    finally:
        db.close()


def save_order(restaurant_id, restaurant_name, caller, call_sid, normalized, raw_json):
    db = SessionLocal()
    try:
        db.add(models.Order(
            restaurant_id=restaurant_id,
            restaurant=restaurant_name,
            caller=caller,
            call_sid=call_sid,
            order_type="reservation",
            full_name=normalized["full_name"],
            address=normalized["date-arrival"],
            house_number=normalized["time_arrival"],
            ordered_items=str(normalized["total_peoples"]),
            payment_method=normalized["contact_number"],
            status="completed",
            raw_json=raw_json,
        ))
        db.commit()
        logger.info(
            "[ORDER] ✅ SAVED  name=%r  date=%r  time=%r  people=%s  phone=%r  restaurant=%s  sid=%s",
            normalized["full_name"], normalized["date-arrival"], normalized["time_arrival"],
            normalized["total_peoples"], normalized["contact_number"],
            restaurant_name, call_sid or "n/a",
        )
    except Exception:
        logger.exception("[ORDER] ❌ DB save FAILED")
        raise
    finally:
        db.close()


# ── OpenAI helpers ─────────────────────────────────────────────────────────────
def log_openai_status(ctx: str, ok: bool, detail: str = ""):
    print(f"[OPENAI][{ctx}] {'WORKING' if ok else 'NOT WORKING'}{' | ' + detail if detail else ''}")


def extract_openai_error(body, fallback: str = "") -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    return fallback[:300] if fallback else "Unknown error"


def check_openai_connectivity():
    try:
        r = requests.get(
            "https://api.openai.com/v1/models/gpt-4o-mini",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            timeout=15,
        )
        body = None
        try:
            body = r.json()
        except ValueError:
            pass
        if r.status_code >= 400:
            return False, f"HTTP {r.status_code}: {extract_openai_error(body, r.text)}"
        return True, "API reachable"
    except Exception as exc:
        return False, str(exc)


def complete_twilio_call(call_sid: str) -> bool:
    if not call_sid or not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        return False
    try:
        Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN).calls(call_sid).update(status="completed")
        logger.info("[TWILIO] ✅ Call completed: %s", call_sid)
        return True
    except Exception as exc:
        logger.warning("[TWILIO] ⚠ Could not complete call %s: %s", call_sid, exc)
        return False


# ── Order JSON helpers ─────────────────────────────────────────────────────────
def _clean(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"", "none", "null", "n/a", "na", "unknown"} else s


def _pick(d: dict, *keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def normalize_reservation(data: dict) -> tuple[dict, list[str]]:
    if not isinstance(data, dict):
        return {}, ["full_name", "date-arrival", "time_arrival", "total_peoples", "contact_number"]

    full_name      = _clean(_pick(data, "full_name", "fullName", "name", "customer_name"))
    date_arrival   = _clean(_pick(data, "date-arrival", "date_arrival", "arrival_date"))
    time_arrival   = _clean(_pick(data, "time_arrival", "arrival_time", "timeArrival", "time"))
    contact_number = _clean(_pick(data, "contact_number", "contactNumber", "phone", "phone_number"))

    raw_total     = _pick(data, "total_peoples", "total_people", "people", "party_size")
    total_peoples = 0
    if isinstance(raw_total, int):
        total_peoples = raw_total
    elif raw_total is not None:
        digits = "".join(c for c in str(raw_total) if c.isdigit())
        if digits:
            total_peoples = int(digits)

    missing = []
    for fname, fval in [("full_name", full_name), ("date-arrival", date_arrival),
                        ("time_arrival", time_arrival), ("contact_number", contact_number)]:
        if not fval:
            missing.append(fname)
    if total_peoples <= 0:
        missing.append("total_peoples")

    return {
        "order_type":     "reservation",
        "full_name":      full_name,
        "date-arrival":   date_arrival,
        "time_arrival":   time_arrival,
        "total_peoples":  total_peoples,
        "contact_number": contact_number,
    }, missing


def parse_order_json(text: str) -> dict | None:
    idx = text.find(ORDER_JSON_MARKER)
    if idx < 0:
        return None

    tail = text[idx + len(ORDER_JSON_MARKER):].lstrip(" \t\n:")
    if tail.startswith("```"):
        nl   = tail.find("\n")
        tail = tail[nl + 1:] if nl != -1 else tail[3:]

    start = tail.find("{")
    if start < 0:
        return None

    depth = 0
    end   = -1
    for i in range(start, len(tail)):
        if tail[i] == "{":
            depth += 1
        elif tail[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        return None

    try:
        return json.loads(tail[start: end + 1])
    except ValueError as exc:
        logger.warning("[ORDER] JSON parse error: %s  raw=%r", exc, tail[start: end + 1][:300])
        return None


def texts_from_event(ev: dict) -> list[str]:
    if not isinstance(ev, dict):
        return []
    if ev.get("type") in {"response.output_audio.delta", "response.audio.delta"}:
        return []

    out: list[str] = []

    def add(v):
        if isinstance(v, str) and v:
            out.append(v)

    for k in ("delta", "text", "transcript"):
        add(ev.get(k))

    for ck in ("item", "response"):
        obj = ev.get(ck)
        if not isinstance(obj, dict):
            continue
        for k in ("text", "transcript"):
            add(obj.get(k))
        for part in obj.get("content") or []:
            if isinstance(part, dict):
                for k in ("text", "transcript"):
                    add(part.get(k))

    if ev.get("type") == "response.done":
        for item in (ev.get("response") or {}).get("output") or []:
            if not isinstance(item, dict):
                continue
            for part in item.get("content") or []:
                if isinstance(part, dict):
                    for k in ("text", "transcript"):
                        add(part.get(k))

    return out


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    ok, detail = check_openai_connectivity()
    log_openai_status("startup", ok, detail)
    ensure_admin_user()


@app.get("/health/openai", response_class=JSONResponse)
@app.get("/api/health/openai", response_class=JSONResponse)
async def health_openai():
    ok, detail = check_openai_connectivity()
    log_openai_status("health", ok, detail)
    return {"working": ok, "detail": detail}


@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"status": "running"}


@app.get("/dashboard")
@app.get("/dashboard.html")
async def dashboard_page():
    if not DASHBOARD_FILE.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return FileResponse(DASHBOARD_FILE)


# ── Twilio webhook ─────────────────────────────────────────────────────────────
@app.api_route("/incoming-call",                     methods=["GET", "POST"])
@app.api_route("/incoming-call/{restaurant_id}",     methods=["GET", "POST"])
@app.api_route("/api/incoming-call",                 methods=["GET", "POST"])
@app.api_route("/api/incoming-call/{restaurant_id}", methods=["GET", "POST"])
async def incoming_call(request: Request, restaurant_id: int | None = None):
    if restaurant_id is None:
        q = request.query_params.get("restaurant_id")
        if q and q.isdigit():
            restaurant_id = int(q)

    p        = dict(request.query_params)
    caller   = str(p.get("From") or p.get("Caller") or "Unknown")
    call_sid = str(p.get("CallSid") or "")

    response = VoiceResponse()
    if restaurant_id is not None and not restaurant_exists_and_active(restaurant_id):
        response.say("Invalid restaurant. Please contact support.", voice="alice")
        response.append(Hangup())
        return HTMLResponse(content=str(response), media_type="application/xml")

    host      = request.headers.get("host")
    path      = request.url.path or ""
    root_path = request.scope.get("root_path") or ""
    prefix    = root_path or ("/api" if path.startswith("/api/") else "")
    qs        = f"edge=dublin&caller={quote_plus(caller)}&call_sid={quote_plus(call_sid)}"
    ws_url    = (
        f"wss://{host}{prefix}/media-stream/{restaurant_id}?{qs}"
        if restaurant_id is not None
        else f"wss://{host}{prefix}/media-stream?{qs}"
    )

    connect = Connect()
    connect.stream(url=ws_url)
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")


# ── REST endpoints ─────────────────────────────────────────────────────────────
@app.post("/chat")
@app.post("/api/chat")
async def chat_with_ai(data: dict, user=Depends(require_auth)):
    restaurant_id = data.get("restaurant_id")
    r_prompt      = data.get("restaurant_prompt")
    if not user.is_admin:
        if not user.restaurant_id:
            raise HTTPException(status_code=403, detail="Forbidden")
        restaurant_id = user.restaurant_id

    final_prompt = compose_system_prompt(r_prompt)
    if not (r_prompt or "").strip() and restaurant_id is not None:
        try:
            final_prompt = get_restaurant_prompt(int(restaurant_id))
        except (TypeError, ValueError):
            pass

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": final_prompt},
                    {"role": "user",   "content": data.get("message", "")},
                ],
            },
        )

    body = {}
    try:
        body = r.json()
    except ValueError:
        pass
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"OpenAI: {extract_openai_error(body, r.text)}")
    choices = body.get("choices") or []
    reply   = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
    return {"reply": str(reply)}


@app.post("/orders/ingest")
@app.post("/api/orders/ingest")
async def ingest_order(data: dict, user=Depends(require_auth)):
    restaurant_id = data.get("restaurant_id")
    if not user.is_admin:
        if not user.restaurant_id:
            raise HTTPException(status_code=403, detail="Forbidden")
        restaurant_id = user.restaurant_id
    try:
        restaurant_id = int(restaurant_id) if restaurant_id is not None else None
    except (TypeError, ValueError):
        restaurant_id = None

    normalized, missing = normalize_reservation(data)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing: {', '.join(missing)}")

    raw_json = data.get("raw_json") or json.dumps(
        {k: normalized[k] for k in ("full_name", "date-arrival", "time_arrival", "total_peoples", "contact_number")}
    )
    try:
        save_order(
            restaurant_id, get_restaurant_name(restaurant_id),
            str(data.get("caller") or "web"), str(data.get("call_sid") or ""),
            normalized, raw_json,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="order save failed")
    return {"ok": True}


@app.api_route("/voice-session", methods=["GET", "POST"])
@app.api_route("/api/voice-session", methods=["GET", "POST"])
def create_voice_session(request: Request, data: dict | None = None, user=Depends(require_auth)):
    data          = data or {}
    restaurant_id = data.get("restaurant_id") or request.query_params.get("restaurant_id")
    r_prompt      = data.get("restaurant_prompt")
    if not user.is_admin:
        if not user.restaurant_id:
            raise HTTPException(status_code=403, detail="Forbidden")
        restaurant_id = user.restaurant_id

    final_prompt = compose_system_prompt(r_prompt)
    if not (r_prompt or "").strip() and restaurant_id is not None:
        try:
            final_prompt = get_restaurant_prompt(int(restaurant_id))
        except (TypeError, ValueError):
            pass

    r = requests.post(
        "https://api.openai.com/v1/realtime/sessions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={"model": REALTIME_MODEL, "voice": VOICE, "instructions": final_prompt},
    )
    try:
        body = r.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="Invalid JSON from OpenAI")
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"OpenAI: {extract_openai_error(body, r.text)}")
    if not isinstance(body, dict) or not (body.get("client_secret") or {}).get("value"):
        raise HTTPException(status_code=502, detail="Missing client_secret")
    body["model"] = REALTIME_MODEL
    return body


# ── Realtime session init ──────────────────────────────────────────────────────
async def init_session(openai_ws, instructions: str):
    await openai_ws.send(json.dumps({
        "type": "session.update",
        "session": {
            "model":                     REALTIME_MODEL,
            "instructions":              instructions,
            "modalities":                ["audio", "text"],
            "voice":                     VOICE,
            "input_audio_format":        "g711_ulaw",
            "output_audio_format":       "g711_ulaw",
            "input_audio_transcription": {"model": "whisper-1"},
            "turn_detection": {
                "type":                "server_vad",
                "threshold":           TURN_DETECTION_THRESHOLD,
                "silence_duration_ms": TURN_DETECTION_SILENCE_MS,
                "prefix_padding_ms":   TURN_DETECTION_PREFIX_MS,
                "create_response":     TURN_DETECTION_CREATE_RESPONSE,
                "interrupt_response":  TURN_DETECTION_INTERRUPT_RESPONSE,
            },
        },
    }))
    await openai_ws.send(json.dumps({"type": "response.create"}))


# ── WebSocket bridge ───────────────────────────────────────────────────────────
async def handle_media_stream_with_id(websocket: WebSocket, restaurant_id: int | None):
    await websocket.accept()
    logger.info("[CALL] ▶ New call  restaurant_id=%s", restaurant_id)

    instructions    = get_restaurant_prompt(restaurant_id)
    restaurant_name = get_restaurant_name(restaurant_id)
    caller          = websocket.query_params.get("caller") or "Unknown"
    call_sid        = websocket.query_params.get("call_sid") or ""

    logger.info("[PROMPT] Sending to OpenAI:\n%s", instructions)

    try:
        async with websockets.connect(
            f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}&temperature={TEMPERATURE}",
            additional_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta":   "realtime=v1",
            },
            ping_interval=20,
            ping_timeout=30,
        ) as openai_ws:
            logger.info("[OPENAI] ▶ Realtime WS connected")
            await init_session(openai_ws, instructions)

            stream_sid          = None
            last_assistant_item = None
            response_active     = False
            response_cancelling = False
            start_ts            = None
            logged              = False
            call_ending         = False

            full_text   = ""
            order_saved = False
            arm_hangup  = False

            def log_call():
                nonlocal logged
                if logged:
                    return
                logged   = True
                duration = (datetime.utcnow() - start_ts).total_seconds() if start_ts else 0
                save_call_log(
                    restaurant_id, restaurant_name, caller, float(duration),
                    "missed" if duration < 3 else "completed",
                )

            async def oai(msg: dict):
                try:
                    await openai_ws.send(json.dumps(msg))
                except Exception as e:
                    logger.debug("[OAI-SEND] %s", e)

            async def twi(msg: dict):
                try:
                    await websocket.send_json(msg)
                except Exception as e:
                    logger.debug("[TWI-SEND] %s", e)

            async def interrupt():
                nonlocal response_active, response_cancelling, last_assistant_item
                if response_cancelling:
                    return
                response_cancelling = True
                response_active     = False
                last_assistant_item = None
                logger.info("[CALL] ✂ Interrupting")
                await oai({"type": "response.cancel"})
                if stream_sid:
                    await twi({"event": "clear", "streamSid": stream_sid})

            async def end_call(reason=""):
                nonlocal call_ending
                if call_ending:
                    return
                call_ending = True
                logger.info("[CALL] ◼ Hanging up — %s", reason)
                if call_sid:
                    await asyncio.to_thread(complete_twilio_call, call_sid)
                for fn in (websocket.close, openai_ws.close):
                    try:
                        await fn()
                    except Exception:
                        pass

            def try_save() -> bool:
                nonlocal order_saved
                if order_saved:
                    return True
                if ORDER_JSON_MARKER not in full_text:
                    return False

                mi      = full_text.rfind(ORDER_JSON_MARKER)
                context = full_text[max(0, mi - 20): mi + 500]
                logger.info("[ORDER] 🔍 Marker found:\n%r", context)

                parsed = parse_order_json(full_text)
                if parsed is None:
                    logger.info("[ORDER] ⏳ JSON incomplete — waiting")
                    return False

                logger.info("[ORDER] 📦 Parsed: %s", json.dumps(parsed, ensure_ascii=False))

                normalized, missing = normalize_reservation(parsed)
                if missing:
                    logger.warning("[ORDER] ⚠ Missing: %s — waiting", missing)
                    return False

                raw_json = json.dumps({
                    "full_name":      normalized["full_name"],
                    "date-arrival":   normalized["date-arrival"],
                    "time_arrival":   normalized["time_arrival"],
                    "total_peoples":  normalized["total_peoples"],
                    "contact_number": normalized["contact_number"],
                })
                try:
                    save_order(restaurant_id, restaurant_name, caller, call_sid, normalized, raw_json)
                    order_saved = True
                    return True
                except Exception:
                    return False

            async def receive_twilio():
                nonlocal stream_sid, start_ts, call_sid
                try:
                    async for message in websocket.iter_text():
                        ev    = json.loads(message)
                        etype = ev.get("event")

                        if etype == "start":
                            stream_sid = ev["start"]["streamSid"]
                            start_ts   = datetime.utcnow()
                            if not call_sid:
                                call_sid = str(ev["start"].get("callSid") or "")
                            logger.info("[CALL] ▶ Stream  sid=%s  caller=%s", stream_sid, caller)

                        elif etype == "media":
                            await oai({
                                "type":  "input_audio_buffer.append",
                                "audio": ev["media"]["payload"],
                            })

                        elif etype == "stop":
                            logger.info("[CALL] ■ Stream stopped  sid=%s", stream_sid)
                            try:
                                await openai_ws.close()
                            except Exception:
                                pass
                            break

                except WebSocketDisconnect:
                    logger.info("[CALL] Twilio disconnected  sid=%s", stream_sid)
                    try:
                        await openai_ws.close()
                    except Exception:
                        pass
                except Exception as exc:
                    s = str(exc)
                    if any(x in s for x in ("WebSocket is not connected", "1000", "1001")):
                        logger.info("[CALL] Twilio WS closed  sid=%s", stream_sid)
                    else:
                        logger.exception("[CALL] receive_twilio: %s", exc)
                    try:
                        await openai_ws.close()
                    except Exception:
                        pass

            async def send_twilio():
                nonlocal last_assistant_item, response_active, response_cancelling
                nonlocal full_text, order_saved, arm_hangup

                async for raw in openai_ws:
                    ev    = json.loads(raw)
                    etype = ev.get("type", "")

                    if REALTIME_LOG_EVERY_EVENT:
                        logger.info("[OAI-EVENT] %s", etype)

                    if etype == "response.created":
                        response_active     = True
                        response_cancelling = False

                    elif etype in {"response.done", "response.cancelled"}:
                        response_active     = False
                        response_cancelling = False
                        last_assistant_item = None

                        if arm_hangup and not call_ending:
                            logger.info("[CALL] ✅ Confirmation done — hanging up in 1s")
                            await asyncio.sleep(1.0)
                            await end_call("auto-hangup after confirmation")
                            break

                    elif etype == "error":
                        err  = ev.get("error") or {}
                        code = err.get("code", "")
                        if code in ("response_cancel_not_active", "response_already_cancelled",
                                    "item_truncated"):
                            response_cancelling = False
                            continue
                        logger.error("[OAI] ❌ %s: %s", code, err.get("message", ""))
                        continue

                    # Accumulate all text
                    new_texts = texts_from_event(ev)
                    if new_texts:
                        chunk = "".join(new_texts)
                        full_text += chunk
                        if len(full_text) > 60_000:
                            full_text = full_text[-30_000:]
                        if chunk.strip():
                            logger.info("[MODEL-TEXT] %s", chunk)

                    # Try save on every chunk
                    if not order_saved and ORDER_JSON_MARKER in full_text:
                        if try_save():
                            logger.info("[ORDER] ✅ Saved — arming hangup")
                            arm_hangup = True

                    # Interrupt only before order saved
                    if etype == "input_audio_buffer.speech_started":
                        if not order_saved:
                            logger.info("[CALL] 🗣 Interrupting")
                            await interrupt()
                        else:
                            logger.info("[CALL] 🗣 Ignored (post-order)")

                    elif etype == "input_audio_buffer.speech_stopped":
                        if not response_active and not response_cancelling:
                            await oai({"type": "response.create"})
                            response_active = True

                    if etype in {"response.output_audio.delta", "response.audio.delta"}:
                        audio = ev.get("delta")
                        if isinstance(audio, str) and stream_sid:
                            await twi({
                                "event":     "media",
                                "streamSid": stream_sid,
                                "media":     {"payload": audio},
                            })
                        if ev.get("item_id"):
                            last_assistant_item = ev["item_id"]
                            response_active     = True

            try:
                await asyncio.gather(receive_twilio(), send_twilio())
            finally:
                log_call()

    except websockets.exceptions.ConnectionClosedError as exc:
        logger.warning("[OPENAI] WS closed: %s", exc)
        try:
            await websocket.close()
        except Exception:
            pass
    except Exception:
        logger.exception("[CALL] Bridge crashed  restaurant_id=%s  caller=%s", restaurant_id, caller)
        try:
            await websocket.close()
        except Exception:
            pass


# ── WebSocket routes ───────────────────────────────────────────────────────────
@app.websocket("/media-stream")
@app.websocket("/api/media-stream")
async def handle_media_stream(websocket: WebSocket):
    q = websocket.query_params.get("restaurant_id")
    await handle_media_stream_with_id(websocket, int(q) if q and q.isdigit() else None)


@app.websocket("/media-stream/{restaurant_id}")
@app.websocket("/api/media-stream/{restaurant_id}")
async def handle_media_stream_restaurant(websocket: WebSocket, restaurant_id: int):
    await handle_media_stream_with_id(websocket, restaurant_id)


@app.get("/media-stream")
@app.get("/media-stream/{restaurant_id}")
@app.get("/api/media-stream")
@app.get("/api/media-stream/{restaurant_id}")
def media_stream_http_guard(restaurant_id: int | None = None):
    return JSONResponse(status_code=426, content={
        "error":  "WebSocket required",
        "detail": "Use /incoming-call Twilio webhook to start a media stream.",
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
