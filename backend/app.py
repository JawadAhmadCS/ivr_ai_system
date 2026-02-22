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
from twilio.twiml.voice_response import VoiceResponse, Connect, Hangup

from database import engine, SessionLocal, ensure_schema
import models
from routes import restaurant, call_logs, dashboard, prompt, auth as auth_routes
from auth import ensure_admin_user, require_auth

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PORT = int(os.getenv("PORT", 5050))
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-realtime")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))
VOICE = os.getenv("VOICE", "alloy")

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
    return f"{global_prompt}\n{addon}" if addon else global_prompt


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
                    "instructions": "ברכי את המתקשר ושאלי איך אפשר לעזור לו היום, בעברית.",
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

    async with websockets.connect(
        f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}&temperature={TEMPERATURE}",
        additional_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
    ) as openai_ws:
        await init_session(openai_ws, instructions)

        stream_sid = None
        last_assistant_item = None
        start_ts = None
        logged = False

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

        async def send_twilio():
            nonlocal last_assistant_item

            async for msg in openai_ws:
                response = json.loads(msg)

                if response.get("type") == "response.output_audio.delta":
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

                if response.get("type") == "input_audio_buffer.speech_started":
                    if last_assistant_item:
                        print("Interrupt detected")

                        await openai_ws.send(json.dumps({"type": "response.cancel"}))
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
