import os
import json
import asyncio
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

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PORT = int(os.getenv("PORT", 5050))
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))
VOICE = os.getenv("VOICE", "alloy")
LLM_HANGUP_TOKEN = (os.getenv("LLM_HANGUP_TOKEN") or "<hangup>").strip()
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
ORDER_JSON_MARKER = "ORDER_JSON:"
ORDER_TYPE_VALUES = {"delivery", "pickup"}
PAYMENT_METHOD_VALUES = {"card", "cash"}
ORDER_CAPTURE_RULES = (
    "Order capture requirements:\n"
    "- Collect order_type (delivery or pickup), full_name, ordered_items, and payment_method (card or cash).\n"
    "- If order_type is delivery, also collect address and house_number.\n"
    "- Do not complete the order until all required fields are provided.\n"
    "- When complete, output a single-line JSON object prefixed with ORDER_JSON: using ONLY these keys:\n"
    '  order_type, full_name, address, house_number, ordered_items, payment_method.\n'
    "- Use valid JSON with double quotes. Do not read the JSON aloud to the caller."
)

BASE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = BASE_DIR / "prompts"
RESTAURANT_FILE = PROMPT_DIR / "restaurants.json"


app = FastAPI()

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
    hangup_rule = "Only output `<hangup>` when the user clearly asks to end/disconnect the call now. Otherwise never output `<hangup>`."
    base = f"{global_prompt}\n{addon}" if addon else global_prompt
    return f"{base}\n{hangup_rule}\n{ORDER_CAPTURE_RULES}"


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


def save_call_log(restaurant_id: int | None, restaurant_name: str, caller: str, duration: float, status: str):
    db = SessionLocal()
    try:
        db.add(
            models.CallLog(
                restaurant_id=restaurant_id,
                restaurant=restaurant_name,
                caller=caller,
                duration=duration,
                status=status,
            )
        )
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


def complete_twilio_call(call_sid: str) -> bool:
    if not call_sid:
        return False
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        print("[TWILIO] Missing TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN; skipping REST hangup")
        return False
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.calls(call_sid).update(status="completed")
        print(f"[TWILIO] Call completed via REST API: {call_sid}")
        return True
    except Exception as e:
        print(f"[TWILIO] Failed to complete call {call_sid}: {e}")
        return False


def extract_text_candidates(response: dict) -> list[str]:
    if not isinstance(response, dict):
        return []

    event_type = str(response.get("type", ""))
    if event_type == "response.output_audio.delta":
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
        return {}, ["order_type", "full_name", "ordered_items", "payment_method"]

    order_type_raw = _clean_field(
        _first_value(data, ["order_type", "orderType", "type", "orderType", "order"])
    ).lower()
    if "deliver" in order_type_raw:
        order_type = "delivery"
    elif "pick" in order_type_raw or "take" in order_type_raw:
        order_type = "pickup"
    else:
        order_type = order_type_raw

    payment_raw = _clean_field(
        _first_value(data, ["payment_method", "paymentMethod", "payment", "payment_type"])
    ).lower()
    if "card" in payment_raw or "credit" in payment_raw or "debit" in payment_raw:
        payment_method = "card"
    elif "cash" in payment_raw:
        payment_method = "cash"
    else:
        payment_method = payment_raw

    full_name = _clean_field(
        _first_value(data, ["full_name", "fullName", "name", "customer_name", "customerName"])
    )
    address = _clean_field(_first_value(data, ["address", "street", "street_address", "streetAddress"]))
    house_number = _clean_field(
        _first_value(data, ["house_number", "houseNo", "house_no", "house", "apt", "apartment"])
    )

    items_value = _first_value(data, ["ordered_items", "items", "order_items", "items_list", "orderItems"])
    if isinstance(items_value, list):
        parts = []
        for v in items_value:
            if isinstance(v, dict):
                name = v.get("name") or v.get("item") or v.get("title")
                qty = v.get("qty") or v.get("quantity")
                if name and qty:
                    parts.append(f"{name} x{qty}")
                elif name:
                    parts.append(str(name))
                else:
                    parts.append(str(v))
            else:
                cleaned = _clean_field(v)
                if cleaned:
                    parts.append(cleaned)
        ordered_items = ", ".join(p for p in parts if p)
    else:
        ordered_items = _clean_field(items_value)

    missing = []
    if order_type not in ORDER_TYPE_VALUES:
        missing.append("order_type")
    if not full_name:
        missing.append("full_name")
    if not ordered_items:
        missing.append("ordered_items")
    if payment_method not in PAYMENT_METHOD_VALUES:
        missing.append("payment_method")
    if order_type == "delivery":
        if not address:
            missing.append("address")
        if not house_number:
            missing.append("house_number")

    normalized = {
        "order_type": order_type,
        "full_name": full_name,
        "address": address,
        "house_number": house_number,
        "ordered_items": ordered_items,
        "payment_method": payment_method,
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

    json_str = tail[start : end + 1]
    try:
        data = json.loads(json_str)
    except ValueError:
        return None
    return json_str, data


def save_order(
    restaurant_id: int | None,
    restaurant_name: str,
    caller: str,
    call_sid: str,
    normalized: dict,
    raw_json: str,
    status: str = "completed",
):
    db = SessionLocal()
    try:
        db.add(
            models.Order(
                restaurant_id=restaurant_id,
                restaurant=restaurant_name,
                caller=caller,
                call_sid=call_sid,
                order_type=normalized.get("order_type"),
                full_name=normalized.get("full_name"),
                address=normalized.get("address"),
                house_number=normalized.get("house_number"),
                ordered_items=normalized.get("ordered_items"),
                payment_method=normalized.get("payment_method"),
                status=status,
                raw_json=raw_json,
            )
        )
        db.commit()
    finally:
        db.close()


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


@app.api_route("/incoming-call", methods=["GET", "POST"])
@app.api_route("/incoming-call/{restaurant_id}", methods=["GET", "POST"])
async def incoming_call(request: Request, restaurant_id: int | None = None):
    response = VoiceResponse()

    if restaurant_id is None:
        q_id = request.query_params.get("restaurant_id")
        if q_id and q_id.isdigit():
            restaurant_id = int(q_id)

    params = dict(request.query_params)
    caller = params.get("From") or params.get("Caller") or "Unknown"
    caller = str(caller) if caller is not None else "Unknown"
    call_sid = params.get("CallSid") or ""
    call_sid = str(call_sid) if call_sid is not None else ""

    if restaurant_id is not None and not restaurant_exists_and_active(restaurant_id):
        response.say("Invalid restaurant id. Please contact support.", voice="alice")
        response.append(Hangup())
        return HTMLResponse(content=str(response), media_type="application/xml")

    # response.say("Hello, how can I help you?", voice="alice")

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

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": final_prompt},
            {"role": "user", "content": user_msg},
        ],
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
        )

    try:
        body = r.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="Invalid response from OpenAI API")

    if r.status_code >= 400:
        error_message = ""
        if isinstance(body, dict):
            error_message = body.get("error", {}).get("message", "")
        if not error_message:
            error_message = r.text[:300]
        log_openai_status("chat", False, f"HTTP {r.status_code}: {error_message}")
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI API error ({r.status_code}): {error_message}",
        )

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
            print(f"[ORDER] ingest missing fields: {missing} | keys={list(data.keys())}")
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {', '.join(missing)}",
            )

        restaurant_name = get_restaurant_name(restaurant_id)
        caller = str(data.get("caller") or "web")
        call_sid = str(data.get("call_sid") or "")
        raw_json = data.get("raw_json")
        if not isinstance(raw_json, str) or not raw_json.strip():
            raw_json = json.dumps(
                {
                    "order_type": normalized.get("order_type"),
                    "full_name": normalized.get("full_name"),
                    "address": normalized.get("address"),
                    "house_number": normalized.get("house_number"),
                    "ordered_items": normalized.get("ordered_items"),
                    "payment_method": normalized.get("payment_method"),
                }
            )

        save_order(
            restaurant_id=restaurant_id,
            restaurant_name=restaurant_name,
            caller=caller,
            call_sid=call_sid,
            normalized=normalized,
            raw_json=raw_json,
            status="completed",
        )
        print(f"[ORDER] Web order saved for {restaurant_name} ({caller})")
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
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": REALTIME_MODEL,
            "voice": VOICE,
            "instructions": final_prompt,
        }

        r = requests.post(url, headers=headers, json=payload)
        try:
            body = r.json()
        except ValueError:
            log_openai_status("voice-session", False, "Invalid JSON from OpenAI realtime")
            raise HTTPException(status_code=502, detail="Invalid response from OpenAI realtime API")

        if r.status_code >= 400:
            error_message = extract_openai_error(body, r.text)
            log_openai_status("voice-session", False, f"HTTP {r.status_code}: {error_message}")
            raise HTTPException(
                status_code=502,
                detail=f"OpenAI realtime error ({r.status_code}): {error_message}",
            )

        if not isinstance(body, dict) or not body.get("client_secret", {}).get("value"):
            log_openai_status("voice-session", False, "Missing client_secret in realtime response")
            raise HTTPException(status_code=502, detail="OpenAI realtime response missing client_secret")

        body["model"] = REALTIME_MODEL
        log_openai_status("voice-session", True, "session token issued")
        return body

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(e)
        raise HTTPException(status_code=500, detail="session error")


async def init_session(openai_ws, instructions: str):
    session = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": REALTIME_MODEL,
            "instructions": instructions,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "noise_reduction": {"type": "near_field"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.4,
                        "silence_duration_ms": 300,
                        "prefix_padding_ms": 150,
                    },
                },
                "output": {
                    "format": {"type": "audio/pcmu"},
                    "voice": VOICE,
                    "speed": 1.05,
                },
            },
        },
    }
    await openai_ws.send(json.dumps(session))
    # Kick off with a greeting from the assistant
    await openai_ws.send(
        json.dumps(
            {
                "type": "response.create",
                "response": {
                    "output_modalities": ["audio"],
                },
            }
        )
    )


async def handle_media_stream_with_id(websocket: WebSocket, restaurant_id: int | None):
    await websocket.accept()
    print("Twilio connected")

    instructions = get_restaurant_prompt(restaurant_id)
    restaurant_name = get_restaurant_name(restaurant_id)
    caller = websocket.query_params.get("caller") or "Unknown"
    call_sid = websocket.query_params.get("call_sid") or ""

    async with websockets.connect(
        f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}&temperature={TEMPERATURE}",
        additional_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
    ) as openai_ws:
        await init_session(openai_ws, instructions)

        stream_sid = None
        last_assistant_item = None
        response_active = False
        start_ts = None
        logged = False
        call_already_ending = False
        order_buffer = ""
        order_saved = False
        marker_tail = ""

        def log_call():
            nonlocal logged
            if logged:
                return
            logged = True
            end_ts = datetime.utcnow()
            duration = (end_ts - start_ts).total_seconds() if start_ts else 0
            status = "missed" if duration < 3 else "completed"
            save_call_log(restaurant_id, restaurant_name, caller, float(duration), status)

        async def receive_twilio():
            nonlocal stream_sid
            nonlocal start_ts
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)

                    if data["event"] == "start":
                        stream_sid = data["start"]["streamSid"]
                        start_ts = datetime.utcnow()
                        print("Stream started:", stream_sid)

                    elif data["event"] == "media":
                        await openai_ws.send(
                            json.dumps(
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": data["media"]["payload"],
                                }
                            )
                        )

                    elif data["event"] == "stop":
                        print("Call ended")
                        await openai_ws.close()
                        break

            except WebSocketDisconnect:
                print("Twilio disconnected")
                await openai_ws.close()
            except RuntimeError as e:
                if "WebSocket is not connected" in str(e):
                    print("Twilio websocket already closed before/while reading")
                    await openai_ws.close()
                    return
                raise

        async def end_call_from_assistant():
            nonlocal call_already_ending
            if call_already_ending:
                return
            call_already_ending = True
            print(f"[CALL] Assistant emitted hangup token {LLM_HANGUP_TOKEN!r}; ending call")
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

        async def send_twilio():
            nonlocal last_assistant_item
            nonlocal response_active
            nonlocal order_buffer
            nonlocal order_saved
            nonlocal marker_tail

            async for msg in openai_ws:
                response = json.loads(msg)
                event_type = response.get("type")

                if event_type == "response.created":
                    response_active = True
                elif event_type == "response.done":
                    response_active = False
                    last_assistant_item = None

                if event_type == "error":
                    err = response.get("error") if isinstance(response.get("error"), dict) else {}
                    code = err.get("code", "unknown")
                    message = err.get("message", "unknown error")
                    if code == "response_cancel_not_active":
                        # Benign race when user starts speaking after assistant already ended output.
                        continue
                    print(
                        f"[OPENAI][realtime][error] {code}: {message}"
                    )
                    continue

                if not order_saved:
                    text_candidates = extract_text_candidates(response)
                    for text in text_candidates:
                        if order_buffer:
                            order_buffer += text
                        else:
                            combined = marker_tail + text
                            if ORDER_JSON_MARKER not in combined:
                                marker_tail = combined[-(len(ORDER_JSON_MARKER) - 1):]
                                continue
                            idx = combined.find(ORDER_JSON_MARKER)
                            order_buffer = combined[idx:]
                        if len(order_buffer) > 10000:
                            order_buffer = order_buffer[-10000:]
                        parsed = parse_order_json_from_buffer(order_buffer)
                        if not parsed:
                            continue
                        raw_json, data = parsed
                        normalized, missing = normalize_order_payload(data)
                        if missing:
                            print(f"[ORDER] Incomplete order JSON; missing: {missing}")
                            order_buffer = ""
                            marker_tail = ""
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
                            print(f"[ORDER] Order saved for {restaurant_name} ({call_sid or 'no call sid'})")
                        except Exception as e:
                            print(f"[ORDER] Failed to save order: {e}")
                        break

                if contains_hangup_token(response, LLM_HANGUP_TOKEN):
                    if response_active:
                        await openai_ws.send(json.dumps({"type": "response.cancel"}))
                        response_active = False
                    if stream_sid:
                        await websocket.send_json(
                            {
                                "event": "clear",
                                "streamSid": stream_sid,
                            }
                        )
                    await end_call_from_assistant()
                    break

                if event_type in {"response.output_audio.delta", "response.audio.delta"}:
                    audio = response["delta"]
                    await websocket.send_json(
                        {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": audio},
                        }
                    )

                    if response.get("item_id"):
                        last_assistant_item = response["item_id"]
                        response_active = True

                if event_type == "input_audio_buffer.speech_started":
                    if last_assistant_item and response_active:
                        print("Interrupt detected")

                        await openai_ws.send(json.dumps({"type": "response.cancel"}))
                        response_active = False
                        await websocket.send_json(
                            {
                                "event": "clear",
                                "streamSid": stream_sid,
                            }
                        )

        try:
            await asyncio.gather(receive_twilio(), send_twilio())
        finally:
            log_call()


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
            "error": "WebSocket required",
            "detail": "Use the Twilio Voice webhook /incoming-call to initiate a WebSocket media stream.",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
