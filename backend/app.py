import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream, Hangup
from dotenv import load_dotenv
import httpx
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
from pathlib import Path
from database import engine, SessionLocal
import models
from routes import restaurant, call_logs, dashboard, prompt

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = int(os.getenv('PORT', 5050))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.3))
REALTIME_MODEL = os.getenv('REALTIME_MODEL', 'gpt-realtime')
# "Always respond in Hebrew & instantly. "
BASE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = BASE_DIR / "prompts"
GLOBAL_FILE = PROMPT_DIR / "global_prompt.txt"
RESTAURANT_FILE = PROMPT_DIR / "restaurants.json"

def load_global_prompt():
    if GLOBAL_FILE.exists():
        return GLOBAL_FILE.read_text(encoding="utf-8")
    return "You are a restaurant AI assistant."

def load_restaurants():
    if RESTAURANT_FILE.exists():
        with open(RESTAURANT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


VOICE = 'alloy'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created', 'session.updated'
]
SHOW_TIMING_MATH = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models.Base.metadata.create_all(bind=engine)

app.include_router(restaurant.router)
app.include_router(call_logs.router)
app.include_router(dashboard.router)
app.include_router(prompt.router)

if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')


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


@app.get("/health/openai", response_class=JSONResponse)
async def health_openai():
    ok, detail = check_openai_connectivity()
    log_openai_status("health", ok, detail)
    return {"working": ok, "detail": detail}

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}

def get_restaurant_prompt(restaurant_id: int | None) -> str:
    global_prompt = load_global_prompt()
    if not restaurant_id:
        return global_prompt
    db = SessionLocal()
    try:
        r = db.get(models.Restaurant, restaurant_id)
        if not r or not r.active:
            return global_prompt
        if r.ivr_text and r.ivr_text.strip():
            return f"{global_prompt}\n{r.ivr_text.strip()}"
        return global_prompt
    finally:
        db.close()

def restaurant_exists_and_active(restaurant_id: int) -> bool:
    db = SessionLocal()
    try:
        r = db.get(models.Restaurant, restaurant_id)
        return bool(r and r.active)
    finally:
        db.close()

@app.api_route("/incoming-call", methods=["GET", "POST"])
@app.api_route("/incoming-call/{restaurant_id}", methods=["GET", "POST"])
async def handle_incoming_call(request: Request, restaurant_id: int | None = None):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    # # # <Say> punctuation to improve text-to-speech flow
    # response.say(
    #     "Please wait while we connect",
    #     voice="Google.en-US-Chirp3-HD-Aoede"
    # )
    # response.pause(length=1)
    # response.say(   
    #     "O.K. you can start talking!",
    #     voice="Google.en-US-Chirp3-HD-Aoede"
    # )
    if restaurant_id is None:
        q_id = request.query_params.get("restaurant_id")
        if q_id and q_id.isdigit():
            restaurant_id = int(q_id)

    if restaurant_id is not None and not restaurant_exists_and_active(restaurant_id):
        response.say("Invalid restaurant id. Please contact support.", voice="alice")
        response.append(Hangup())
        return HTMLResponse(content=str(response), media_type="application/xml")

    host = request.url.hostname
    connect = Connect()
    if restaurant_id is not None:
        connect.stream(url=f"wss://{host}/media-stream/{restaurant_id}")
    else:
        connect.stream(url=f"wss://{host}/media-stream")
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.post("/chat")
async def chat_with_ai(data: dict):
    user_msg = data.get("message", "")
    restaurant_prompt = data.get("restaurant_prompt", "")

    global_prompt = load_global_prompt()
    final_prompt = global_prompt + "\n" + restaurant_prompt

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": final_prompt},
            {"role": "user", "content": user_msg}
        ]
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
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
            detail=f"OpenAI API error ({r.status_code}): {error_message}"
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

@app.post("/voice-session")
def create_voice_session(data: dict):
    try:
        restaurant_prompt = data.get("restaurant_prompt", "")
        global_prompt = load_global_prompt()
        final_prompt = global_prompt + "\n" + restaurant_prompt


        url = "https://api.openai.com/v1/realtime/sessions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": REALTIME_MODEL,
            "voice": "alloy",
            "instructions": final_prompt
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
                detail=f"OpenAI realtime error ({r.status_code}): {error_message}"
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

async def handle_media_stream_with_id(websocket: WebSocket, restaurant_id: int | None):
    """Handle WebSocket connections between Twilio and OpenAI."""
    print("Client connected")
    instructions = get_restaurant_prompt(restaurant_id)
    await websocket.accept()

    async with websockets.connect(
        f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}&temperature={TEMPERATURE}",
        additional_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
    ) as openai_ws:
        await initialize_session(openai_ws, instructions)

        # Connection specific state
        stream_sid = None
        latest_media_timestamp = 0
        last_assistant_item = None
        mark_queue = []
        response_start_timestamp_twilio = None
        
        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
            nonlocal stream_sid, latest_media_timestamp
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'media' and openai_ws.state.name == 'OPEN':
                        latest_media_timestamp = int(data['media']['timestamp'])
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    elif data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        print(f"Incoming stream has started {stream_sid}")
                        response_start_timestamp_twilio = None
                        latest_media_timestamp = 0
                        last_assistant_item = None
                    elif data['event'] == 'mark':
                        if mark_queue:
                            mark_queue.pop(0)
            except WebSocketDisconnect:
                print("Client disconnected.")
                if openai_ws.state.name == 'OPEN':
                    await openai_ws.close()

        async def send_to_twilio():
            """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
            nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    if response['type'] in LOG_EVENT_TYPES:
                        print(f"Received event: {response['type']}", response)

                    if response.get('type') == 'response.output_audio.delta' and 'delta' in response:
                        audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                        audio_delta = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": audio_payload
                            }
                        }
                        await websocket.send_json(audio_delta)


                        if response.get("item_id") and response["item_id"] != last_assistant_item:
                            response_start_timestamp_twilio = latest_media_timestamp
                            last_assistant_item = response["item_id"]
                            if SHOW_TIMING_MATH:
                                print(f"Setting start timestamp for new response: {response_start_timestamp_twilio}ms")

                        await send_mark(websocket, stream_sid)

                    # Trigger an interruption. Your use case might work better using `input_audio_buffer.speech_stopped`, or combining the two.
                    if response.get('type') == 'input_audio_buffer.speech_started':
                        print("Speech started detected.")
                        if last_assistant_item:
                            print(f"Interrupting response with id: {last_assistant_item}")
                            await handle_speech_started_event()
            except Exception as e:
                print(f"Error in send_to_twilio: {e}")

        async def handle_speech_started_event():
            """Handle interruption when the caller's speech starts."""
            nonlocal response_start_timestamp_twilio, last_assistant_item
            print("Handling speech started event.")
            if mark_queue and response_start_timestamp_twilio is not None:
                elapsed_time = latest_media_timestamp - response_start_timestamp_twilio
                if SHOW_TIMING_MATH:
                    print(f"Calculating elapsed time for truncation: {latest_media_timestamp} - {response_start_timestamp_twilio} = {elapsed_time}ms")

                if last_assistant_item:
                    if SHOW_TIMING_MATH:
                        print(f"Truncating item with ID: {last_assistant_item}, Truncated at: {elapsed_time}ms")

                    truncate_event = {
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": elapsed_time
                    }
                    await openai_ws.send(json.dumps(truncate_event))

                await websocket.send_json({
                    "event": "clear",
                    "streamSid": stream_sid
                })

                mark_queue.clear()
                last_assistant_item = None
                response_start_timestamp_twilio = None

        async def send_mark(connection, stream_sid):
            if stream_sid:
                mark_event = {
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": "responsePart"}
                }
                await connection.send_json(mark_event)
                mark_queue.append('responsePart')

        await asyncio.gather(receive_from_twilio(), send_to_twilio())

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    q_id = websocket.query_params.get("restaurant_id")
    restaurant_id = int(q_id) if q_id and q_id.isdigit() else None
    await handle_media_stream_with_id(websocket, restaurant_id)

@app.websocket("/media-stream/{restaurant_id}")
async def handle_media_stream_restaurant(websocket: WebSocket, restaurant_id: int):
    await handle_media_stream_with_id(websocket, restaurant_id)

async def send_initial_conversation_item(openai_ws):
    """Send initial conversation item if AI talks first."""
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Say: Hello. How are you?'"
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))


async def initialize_session(openai_ws, instructions: str):
    """Control initial session with OpenAI."""
    session_update = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": REALTIME_MODEL,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "turn_detection": {"type": "server_vad"}
                },
                "output": {
                    "format": {"type": "audio/pcmu"},
                    "voice": VOICE
                }
            },
            "instructions": instructions,
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    # Uncomment the next line to have the AI speak first
    await send_initial_conversation_item(openai_ws)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
