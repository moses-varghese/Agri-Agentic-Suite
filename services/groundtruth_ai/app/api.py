from fastapi import APIRouter
from pydantic import BaseModel
from .services.rag_service import rag_service
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
import os
# Imports for Voice Functionality
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from gtts import gTTS
import json

# --- OFFLINE SPEECH-TO-TEXT SETUP ---
# Point to the model we downloaded in our Dockerfile
VOSK_MODEL_PATH = "/opt/vosk-model"
model = Model(VOSK_MODEL_PATH)

def transcribe_audio_offline(audio_path: str) -> str:
    """
    Transcribes audio to text using the offline Vosk model.
    """
    try:
        # Vosk requires a specific audio format (16-bit PCM, 16000 Hz, mono)
        # We use pydub to convert the uploaded audio to this format
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)

        recognizer = KaldiRecognizer(model, 16000)
        recognizer.AcceptWaveform(audio.raw_data)
        result = json.loads(recognizer.FinalResult())
        
        transcribed_text = result.get('text', '')
        print(f"🎤 Offline Transcription: '{transcribed_text}'")
        return transcribed_text
    except Exception as e:
        print(f"❌ ERROR during transcription: {e}")
        return ""

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@router.post("/generate-truth", response_model=QueryResponse, tags=["RAG Query"])
async def handle_query(request: QueryRequest):
    """
    Endpoint to handle an incoming query. Replaces the /sms endpoint.
    """
    print(f"❓ Received query: '{request.query}'")
    ai_response = await rag_service.get_response(request.query)
    return QueryResponse(answer=ai_response)

@router.post("/voice-query")
async def handle_voice_query(audio_file: UploadFile = File(...)):
    """
    Handles a voice query using a fully offline pipeline.
    """
    # 1. Save uploaded audio file temporarily
    temp_audio_path = f"/tmp/{audio_file.filename}"
    with open(temp_audio_path, "wb") as buffer:
        buffer.write(await audio_file.read())

    # 2. Transcribe audio to text using the OFFLINE model
    query_text = transcribe_audio_offline(temp_audio_path)
    os.remove(temp_audio_path) # Clean up

    if not query_text:
        # Create a generic error audio response
        error_text = "Sorry, I could not understand the audio. Please try again."
        tts = gTTS(error_text, lang='en')
        error_path = "/tmp/error_response.mp3"
        tts.save(error_path)
        return FileResponse(path=error_path, media_type='audio/mpeg', filename='response.mp3')

    # 3. Get text answer from our RAG service
    text_answer = await rag_service.get_response(query_text)

    # 4. Convert text answer to speech (MP3)
    response_path = "/tmp/response.mp3"
    tts = gTTS(text_answer, lang='en')
    tts.save(response_path)

    # 5. Return the audio file
    return FileResponse(path=response_path, media_type='audio/mpeg', filename='response.mp3')