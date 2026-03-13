#!/usr/bin/env python3
"""
SP-LineBot: Production-ready Line Bot with local LLM (Ollama) + Gemini fallback.
Supports multimodal (text, image, voice) with embedding-based RL suggestions.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')
    
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from uvicorn import run as uvicorn_run
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
# Updated LineBot imports for correct message formatting
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, PushMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent, AudioMessageContent

# Import custom modules
from local_llm import OllamaLLMClient, parse_intent, detect_spam
from multimodal import process_image_ocr, process_voice_vosk, extract_metadata_and_embed
from drive_handler import DriveHandler, fetch_user_drive_context
from admin_commands import AdminCommandHandler

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sp_linebot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory
Path('logs').mkdir(exist_ok=True)

# FastAPI app
app = FastAPI(title="SP-LineBot", version="1.0.0")

# ============================================================================
# CLIENT INITIALIZATION
# ============================================================================

# Line Bot credentials
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')

if not LINE_CHANNEL_SECRET or not LINE_CHANNEL_ACCESS_TOKEN:
    raise ValueError("LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN required in .env")

handler = WebhookHandler(LINE_CHANNEL_SECRET)
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)

# Ollama client (local LLM)
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
ollama_client = OllamaLLMClient(host=OLLAMA_HOST)

# Google Drive handler
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON', 'google-service-account.json')
drive_handler = DriveHandler(GOOGLE_SERVICE_ACCOUNT_JSON)

# Admin commands
admin_handler = AdminCommandHandler(drive_handler)

# ============================================================================
# STARTUP & HEALTH CHECK
# ============================================================================

@app.on_event('startup')
async def startup_event():
    """Initialize GPU, warm up Ollama, check timezone."""
    logger.info("=" * 60)
    logger.info("SP-LineBot Starting Up")
    logger.info("=" * 60)
    
    # GPU check
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    logger.info(f"GPU Available: {cuda_available}, Device: {gpu_name}")
    logger.info(f"Torch CUDA: {torch.cuda.is_available()}")
    
    # Ollama warm-up
    try:
        response = await ollama_client.health_check()
        logger.info(f"Ollama Status: {response}")
    except Exception as e:
        logger.warning(f"Ollama not running: {e}. Run 'ollama serve' manually.")
    
    # Dynamic timezone (Bangkok default)
    tz = timezone(timedelta(hours=7))
    now = datetime.now(tz)
    logger.info(f"Timezone: UTC+7 (Bangkok), Current: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Drive handler initialization
    try:
        drive_handler.authenticate()
        logger.info("Google Drive authenticated")
    except Exception as e:
        logger.error(f"Google Drive auth failed: {e}")
    
    logger.info("=" * 60)

@app.get("/health")
async def health_check():
    """Health check endpoint (Ngrok + LB)."""
    cuda_status = torch.cuda.is_available()
    return JSONResponse({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "gpu_available": cuda_status,
        "ollama_enabled": True
    })

# ============================================================================
# WEBHOOK HANDLERS
# ============================================================================

@app.post("/callback")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    """Line Bot webhook receiver."""
    signature = request.headers.get('X-Line-Signature')
    body = await request.body()
    
    try:
        handler.handle(body.decode('utf-8'), signature)
    except InvalidSignatureError:
        logger.error("Invalid signature")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail="Internal error")
    
    return JSONResponse({"status": "ok"})

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):
    """Process text messages with intent chain: Ollama → Gemini → fallback."""
    user_id = event.source.user_id
    text = event.message.text
    
    logger.info(f"Text from {user_id}: {text[:100]}")
    
    # Spam detection
    if detect_spam(text, user_id):
        logger.warning(f"Spam detected from {user_id}")
        return
    
    # Fetch user context (Drive files, history) - properly passing drive_handler
    user_context = fetch_user_drive_context(user_id, drive_handler)
    
    # Intent parsing with Ollama primary
    intent, confidence = parse_intent(text, ollama_client)
    logger.info(f"Intent: {intent} (conf: {confidence:.2f})")
    
    # Generate response chain
    if confidence > 0.7:
        # High confidence - use Ollama intent
        response = handle_intent(intent, user_id, user_context, text)
    else:
        # Low confidence - escalate to Gemini
        response = handle_gemini_escalation(text, user_id)
    
    # Send response using PushMessageRequest
    with ApiClient(configuration) as api_client:
        api = MessagingApi(api_client)
        api.push_message(PushMessageRequest(
            to=user_id,
            messages=[TextMessage(text=response)]
        ))

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event: MessageEvent):
    """Process images: download → OCR (Thai/Eng) → parse → embed."""
    user_id = event.source.user_id
    message_id = event.message.id
    
    logger.info(f"Image from {user_id} (ID: {message_id})")
    
    # Background task
    from fastapi import BackgroundTasks
    background_tasks = BackgroundTasks()
    background_tasks.add_task(process_image_message, user_id, message_id)

def process_image_message(user_id: str, message_id: str):
    """Background: download image, OCR, embed."""
    try:
        with ApiClient(configuration) as api_client:
            api = MessagingApi(api_client)
            content = api.get_message_content(message_id)
        
        # Save image
        img_path = f"uploads/{message_id}.jpg"
        Path('uploads').mkdir(exist_ok=True)
        with open(img_path, 'wb') as f:
            f.write(content.content)
        
        # OCR (Thai + English)
        text, confidence = process_image_ocr(img_path)
        logger.info(f"OCR Result: {text[:100]} (conf: {confidence:.2f})")
        
        # Extract metadata & embed
        extract_metadata_and_embed(img_path, text, user_id)
        
        # Clean temp
        os.remove(img_path)
        
        logger.info(f"Image processed and embedded for {user_id}")
    except Exception as e:
        logger.error(f"Image processing failed: {e}")

@handler.add(MessageEvent, message=AudioMessageContent)
def handle_voice_message(event: MessageEvent):
    """Process voice: download → Vosk STT → intent → respond."""
    user_id = event.source.user_id
    message_id = event.message.id
    duration = event.message.duration
    
    logger.info(f"Voice from {user_id} (duration: {duration}ms)")
    
    # Background processing
    from fastapi import BackgroundTasks
    background_tasks = BackgroundTasks()
    background_tasks.add_task(process_voice_message, user_id, message_id)

def process_voice_message(user_id: str, message_id: str):
    """Background: download audio, STT, intent chain."""
    try:
        with ApiClient(configuration) as api_client:
            api = MessagingApi(api_client)
            content = api.get_message_content(message_id)
        
        # Save audio
        audio_path = f"voice_cache/{message_id}.wav"
        Path('voice_cache').mkdir(exist_ok=True)
        with open(audio_path, 'wb') as f:
            f.write(content.content)
        
        # Vosk STT
        text = process_voice_vosk(audio_path)
        logger.info(f"STT Result: {text}")
        
        # Intent & response (same as text)
        intent, confidence = parse_intent(text, ollama_client)
        response = handle_intent(intent, user_id, {})
        
        # Send response using PushMessageRequest
        with ApiClient(configuration) as api_client:
            api = MessagingApi(api_client)
            api.push_message(PushMessageRequest(
                to=user_id,
                messages=[TextMessage(text=response)]
            ))
        
        # Clean temp
        os.remove(audio_path)
        
        logger.info(f"Voice processed for {user_id}")
    except Exception as e:
        logger.error(f"Voice processing failed: {e}")

# ============================================================================
# INTENT HANDLERS
# ============================================================================

def handle_intent(intent: str, user_id: str, context: Dict[str, Any], text: str = "") -> str:
    """Route to admin commands or general responses."""
    if intent.startswith("ADMIN_"):
        return admin_handler.execute(intent, user_id, text, context)
        
    elif intent == "DRIVE_SCAN":
        # 1. Grab the user's specific Drive folder ID from their context
        folder_id = context.get('drive_folder_id')
        if not folder_id:
            return "You don't have a linked Drive folder yet. Ask an admin to add you."
            
        # 2. Tell the drive_handler to scan the folder and chunk the text
        try:
            count = drive_handler.batch_embed_documents(folder_id, user_id)
            return f"Drive scan complete! Successfully read and memorized {count} text chunks from your files."
        except Exception as e:
            logging.error(f"Drive scan failed: {e}")
            return f"Error scanning drive: {str(e)}"
            
    else:
        # For other intents (like INVENTORY_LOOKUP, etc.)
        return f"Intent: {intent}\nContext: {json.dumps(context, ensure_ascii=False)[:200]}"

def handle_gemini_escalation(text: str, user_id: str) -> str:
    """Fallback to Gemini API when Ollama confidence is low using new google-genai SDK."""
    try:
        from google import genai
        
        # Initialize the new Client
        client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        
        prompt = f"User query (Line Bot): {text}\nProvide concise response (50 words max)."
        
        # Generate content with the new SDK structure
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        logger.info(f"Gemini escalation response: {response.text[:100]}")
        return response.text
    except Exception as e:
        logger.error(f"Gemini escalation failed: {e}")
        return "Sorry, I couldn't process that. Please try again."

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    logger.info(f"Starting FastAPI on 0.0.0.0:{port}")
    uvicorn_run(app, host='0.0.0.0', port=port, log_level='info')