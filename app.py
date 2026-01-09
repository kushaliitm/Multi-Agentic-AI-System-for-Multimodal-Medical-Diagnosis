import os
import uuid
import tempfile
import glob
import threading
import time
from io import BytesIO
from typing import Dict, Union, Optional, List
import logging
import warnings
warnings.filterwarnings("ignore")
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    Cookie,
)
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import requests
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs
from config import Config
from agents.agent_decision import process_query

# -------------------------------------------------------------------
# Application Setup
# -------------------------------------------------------------------

# Configure logging (FastAPI/Uvicorn also use logging, keep formatting consistent)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load application configuration (models, API keys, limits, etc.)
config = Config()

# Initialize FastAPI application
app = FastAPI(
    title="Multi-Agent Medical Chatbot",
    version="2.0",
)

# -------------------------------------------------------------------
# Directory Configuration
# -------------------------------------------------------------------

# File storage locations
UPLOAD_FOLDER = "uploads/backend"              # Temporary backend uploads
FRONTEND_UPLOAD_FOLDER = "uploads/frontend"    # Frontend accessible uploads (if needed)
SPEECH_DIR = "uploads/speech"

# Create required directories on startup
for directory in [UPLOAD_FOLDER, FRONTEND_UPLOAD_FOLDER, SPEECH_DIR]:
    os.makedirs(directory, exist_ok=True)

# -------------------------------------------------------------------
# Static Files and Templates
# -------------------------------------------------------------------

# Serve internal project data and user uploads
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Load Jinja2 templates (index.html UI lives inside templates/)
templates = Jinja2Templates(directory="templates")

# -------------------------------------------------------------------
# Third Party Clients (ElevenLabs)
# -------------------------------------------------------------------

# Initialize ElevenLabs client for transcription (speech-to-text)
client = ElevenLabs(api_key=config.speech.eleven_labs_api_key)

# -------------------------------------------------------------------
# Upload Validation
# -------------------------------------------------------------------

# Allowed image formats for medical image upload endpoints
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename: str) -> bool:
    """
    Validate file extension against supported image formats.

    Args:
        filename: Name of the uploaded file.

    Returns:
        True if file has an allowed image extension, else False.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------------------------------------------------
# Background Cleanup Job
# -------------------------------------------------------------------

def cleanup_old_audio() -> None:
    """
    Periodically delete generated speech files to prevent disk bloat.

    This runs forever in a daemon thread and removes all .mp3 files in SPEECH_DIR
    every 5 minutes.
    """
    while True:
        try:
            files = glob.glob(f"{SPEECH_DIR}/*.mp3")
            for file in files:
                os.remove(file)
            logger.info("Cleaned up old speech files.")
        except Exception as e:
            logger.warning(f"Error during speech cleanup: {e}")
        time.sleep(300)  # run every 5 minutes

# Start cleanup thread as a daemon so it does not block application shutdown
cleanup_thread = threading.Thread(target=cleanup_old_audio, daemon=True)
cleanup_thread.start()

# -------------------------------------------------------------------
# Request Models (Pydantic)
# -------------------------------------------------------------------

class QueryRequest(BaseModel):
    """
    Payload for text-only chat endpoint.

    Attributes:
        query: The user message.
        conversation_history: Optional history (not required if your graph manages state).
    """
    query: str
    conversation_history: List = []

class SpeechRequest(BaseModel):
    """
    Payload for text-to-speech generation.

    Attributes:
        text: Text content to synthesize.
        voice_id: ElevenLabs voice ID to use.
    """
    text: str
    voice_id: str = "EXAMPLE_VOICE_ID"

# -------------------------------------------------------------------
# Routes: UI and Health
# -------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Serve the main web UI (index.html) using Jinja2 templating.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    """
    Health check endpoint (useful for Docker health checks / load balancers).
    """
    return {"status": "healthy"}

# -------------------------------------------------------------------
# Routes: Chat (Text)
# -------------------------------------------------------------------

@app.post("/chat")
def chat(
    request: QueryRequest,
    response: Response,
    session_id: Optional[str] = Cookie(None),
):
    """
    Process a text-only query through the multi-agent orchestration system.

    - Ensures a session_id cookie exists for continuity.
    - Delegates query routing/execution to process_query().
    - Returns final agent response and the agent name used.

    Args:
        request: JSON payload containing user query text.
        response: FastAPI response object to set cookies.
        session_id: Session cookie for user continuity.

    Returns:
        Dict with status, response text, and selected agent name.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        response_data = process_query(request.query)
        response_text = response_data["messages"][-1].content

        # Persist session ID on client
        response.set_cookie(key="session_id", value=session_id)

        result = {
            "status": "success",
            "response": response_text,
            "agent": response_data["agent_name"],
        }


        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# Routes: Upload Image (Medical Vision)
# -------------------------------------------------------------------

@app.post("/upload")
async def upload_image(
    response: Response,
    image: UploadFile = File(...),
    text: str = Form(""),
    session_id: Optional[str] = Cookie(None),
):
    """
    Upload and process a medical image (optionally with text).

    Steps:
    - Validate file extension
    - Validate file size against config.api.max_image_upload_size
    - Save file securely to a backend upload directory
    - Execute agent routing via process_query({"text": ..., "image": ...})
    - Return agent response (and any generated result images)
    - Attempt to delete the temporary uploaded file after processing

    Args:
        response: FastAPI response object to set session cookie
        image: Uploaded image file
        text: Optional user text associated with the image
        session_id: Session cookie for continuity

    Returns:
        Dict with status, response text, agent name, and optionally a result image
    """
    if not allowed_file(image.filename):
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "agent": "System",
                "response": "Unsupported file type. Allowed formats: PNG, JPG, JPEG",
            },
        )

    file_content = await image.read()
    max_bytes = config.api.max_image_upload_size * 1024 * 1024
    if len(file_content) > max_bytes:
        return JSONResponse(
            status_code=413,
            content={
                "status": "error",
                "agent": "System",
                "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB",
            },
        )

    if not session_id:
        session_id = str(uuid.uuid4())

    # Save file with a random UUID prefix to avoid collisions and path traversal
    filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(file_path, "wb") as f:
        f.write(file_content)

    try:
        query = {"text": text, "image": file_path}
        response_data = process_query(query)
        response_text = response_data["messages"][-1].content

        response.set_cookie(key="session_id", value=session_id)

        result = {
            "status": "success",
            "response": response_text,
            "agent": response_data["agent_name"],
        }


        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Best-effort cleanup of uploaded file
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary upload file: {e}")

# -------------------------------------------------------------------
# Routes: Human Validation
# -------------------------------------------------------------------

@app.post("/validate")
def validate_medical_output(
    response: Response,
    validation_result: str = Form(...),
    comments: Optional[str] = Form(None),
    session_id: Optional[str] = Cookie(None),
):
    """
    Handle human-in-the-loop validation for medical AI outputs.

    The validator provides:
    - validation_result: Yes/No
    - optional comments

    The system forwards this feedback back through the agent pipeline so it can:
    - confirm output
    - or generate a safer fallback response

    Args:
        response: FastAPI response object
        validation_result: Form field ('yes' or 'no')
        comments: Optional free-text comments
        session_id: Session cookie for continuity

    Returns:
        Dict indicating validated/rejected status and final response message
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        response.set_cookie(key="session_id", value=session_id)

        validation_query = f"Validation result: {validation_result}"
        if comments:
            validation_query += f" Comments: {comments}"

        response_data = process_query(validation_query)
        final_text = response_data["messages"][-1].content

        if validation_result.lower() == "yes":
            return {
                "status": "validated",
                "message": "**Output confirmed by human validator:**",
                "response": final_text,
            }

        return {
            "status": "rejected",
            "comments": comments,
            "message": "**Output requires further review:**",
            "response": final_text,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# Routes: Speech-to-Text (Transcription)
# -------------------------------------------------------------------

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe uploaded audio using ElevenLabs speech-to-text.

    Workflow:
    - Save .webm temporarily
    - Convert to .mp3 using pydub
    - Send bytes to ElevenLabs speech_to_text endpoint
    - Return transcript text
    - Clean up temporary files

    Returns:
        {"transcript": "..."} on success
    """
    if not audio.filename:
        return JSONResponse(status_code=400, content={"error": "No audio file selected"})

    try:
        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.webm"

        audio_content = await audio.read()
        with open(temp_audio, "wb") as f:
            f.write(audio_content)

        file_size = os.path.getsize(temp_audio)
        logger.info(f"Received audio file size: {file_size} bytes")

        if file_size == 0:
            return JSONResponse(status_code=400, content={"error": "Received empty audio file"})

        mp3_path = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.mp3"

        try:
            audio_segment = AudioSegment.from_file(temp_audio)
            audio_segment.export(mp3_path, format="mp3")

            mp3_size = os.path.getsize(mp3_path)
            logger.info(f"Converted MP3 file size: {mp3_size} bytes")

            with open(mp3_path, "rb") as mp3_file:
                audio_data = mp3_file.read()

            transcription = client.speech_to_text.convert(
                file=audio_data,
                model_id="scribe_v1",
                tag_audio_events=True,
                language_code="eng",
                diarize=True,
            )

            # Cleanup temp files
            try:
                os.remove(temp_audio)
                os.remove(mp3_path)
                logger.info("Deleted temporary transcription files.")
            except Exception as e:
                logger.warning(f"Could not delete temp transcription files: {e}")

            if transcription.text:
                return {"transcript": transcription.text}

            return JSONResponse(
                status_code=500,
                content={"error": "Transcription failed", "details": str(transcription)},
            )

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return JSONResponse(status_code=500, content={"error": f"Error processing audio: {e}"})

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# -------------------------------------------------------------------
# Routes: Text-to-Speech
# -------------------------------------------------------------------

@app.post("/generate-speech")
async def generate_speech(request: SpeechRequest):
    """
    Convert text to speech using ElevenLabs API and return an MP3 file.

    Args:
        request: SpeechRequest containing text and voice_id.

    Returns:
        MP3 audio file response.
    """
    try:
        text = request.text
        selected_voice_id = request.voice_id

        if not text:
            return JSONResponse(status_code=400, content={"error": "Text is required"})

        elevenlabs_url = f"https://api.elevenlabs.io/v1/text-to-speech/{selected_voice_id}/stream"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": config.speech.eleven_labs_api_key,
        }
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
        }

        api_response = requests.post(elevenlabs_url, headers=headers, json=payload)
        if api_response.status_code != 200:
            return JSONResponse(
                status_code=500,
                content={
                    "error": f"Failed to generate speech, status: {api_response.status_code}",
                    "details": api_response.text,
                },
            )

        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio_path = f"./{SPEECH_DIR}/{uuid.uuid4()}.mp3"
        with open(temp_audio_path, "wb") as f:
            f.write(api_response.content)

        return FileResponse(
            path=temp_audio_path,
            media_type="audio/mpeg",
            filename="generated_speech.mp3",
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# -------------------------------------------------------------------
# Exception Handling
# -------------------------------------------------------------------

@app.exception_handler(413)
async def request_entity_too_large(request: Request, exc):
    """
    Central handler for payload-too-large errors (413).
    """
    return JSONResponse(
        status_code=413,
        content={
            "status": "error",
            "agent": "System",
            "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB",
        },
    )

# -------------------------------------------------------------------
# Local Development Entry Point
# -------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host=config.api.host, port=config.api.port)
