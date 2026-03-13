#!/usr/bin/env python3
"""
SP-LineBot: Production-ready Line Bot with local LLM (Ollama) + Gemini fallback.
Supports multimodal (text, image, voice) with RAG (Retrieval-Augmented Generation).
"""

import os
import sys
import threading
from dotenv import load_dotenv

# LOAD ENV VARIABLES FIRST
load_dotenv()

# Create required directories
for d in ['logs', 'uploads', 'voice_cache']:
    os.makedirs(d, exist_ok=True)
    
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from uvicorn import run as uvicorn_run
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, MessagingApiBlob, PushMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent, AudioMessageContent, FileMessageContent

# Import custom modules
from local_llm import OllamaLLMClient, parse_intent, detect_spam
from multimodal import process_image_ocr, process_voice_vosk, extract_metadata_and_embed
from drive_handler import DriveHandler
from admin_commands import AdminCommandHandler

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================

# --- FIX: Enforce UTF-8 for Windows Console and File Logging ---
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sp_linebot.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="SP-LineBot", version="1.0.0")

LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')

if not LINE_CHANNEL_SECRET or not LINE_CHANNEL_ACCESS_TOKEN:
    raise ValueError("LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN required in .env")

handler = WebhookHandler(LINE_CHANNEL_SECRET)
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)

OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
ollama_client = OllamaLLMClient(host=OLLAMA_HOST)

GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON', 'google-service-account.json')
drive_handler = DriveHandler(GOOGLE_SERVICE_ACCOUNT_JSON)
admin_handler = AdminCommandHandler(drive_handler)

# ============================================================================
# STARTUP & HEALTH CHECK
# ============================================================================

@app.on_event('startup')
async def startup_event():
    logger.info("=" * 60)
    logger.info("SP-LineBot Starting Up - Production Mode")
    logger.info("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    logger.info(f"GPU Available: {cuda_available}, Device: {gpu_name}")
    
    try:
        drive_handler.authenticate()
        logger.info("Google Drive authenticated")
    except Exception as e:
        logger.error(f"Google Drive auth failed: {e}")
    
    logger.info("=" * 60)

@app.get("/health")
async def health_check():
    return JSONResponse({"status": "ok", "timestamp": datetime.utcnow().isoformat()})

# ============================================================================
# WEBHOOK HANDLERS
# ============================================================================

@app.post("/callback")
async def webhook(request: Request):
    signature = request.headers.get('X-Line-Signature')
    body = await request.body()
    try:
        handler.handle(body.decode('utf-8'), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return JSONResponse({"status": "ok"})

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):
    user_id = event.source.user_id
    text = event.message.text
    
    logger.info(f"Text from {user_id}: {text[:100]}")
    
    if detect_spam(text, user_id):
        return
    
    user_context = admin_handler.get_user_context(user_id)
    intent, confidence = parse_intent(text, ollama_client)
    logger.info(f"Intent: {intent} (conf: {confidence:.2f})")
    
    if confidence > 0.7:
        response = handle_intent(intent, user_id, user_context, text)
    else:
        response = handle_gemini_escalation(text, user_id)
    
    with ApiClient(configuration) as api_client:
        api = MessagingApi(api_client)
        api.push_message(PushMessageRequest(to=user_id, messages=[TextMessage(text=response)]))

# --- FIX: MULTIMODAL BACKGROUND THREADING ---

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event: MessageEvent):
    user_id = event.source.user_id
    message_id = event.message.id
    logger.info(f"[IMAGE] Image from {user_id} (ID: {message_id})")
    
    # Use standard threading instead of FastAPI BackgroundTasks
    threading.Thread(target=process_image_message, args=(user_id, message_id)).start()

def process_image_message(user_id: str, message_id: str):
    try:
        with ApiClient(configuration) as api_client:
            api_blob = MessagingApiBlob(api_client)
            content = api_blob.get_message_content(message_id)
        
        img_path = f"uploads/{message_id}.jpg"
        with open(img_path, 'wb') as f:
            f.write(content)
        
        text, confidence = process_image_ocr(img_path)
        logger.info(f"OCR Result: {text[:100]} (conf: {confidence:.2f})")
        
        if text.strip():
            extract_metadata_and_embed(img_path, text, user_id)
            msg = f"I successfully read the image! Saved to memory."
        else:
            msg = "I couldn't find any readable text in that image."
            
        with ApiClient(configuration) as api_client:
            api = MessagingApi(api_client)
            api.push_message(PushMessageRequest(to=user_id, messages=[TextMessage(text=msg)]))
            
        os.remove(img_path)
    except Exception as e:
        logger.error(f"Image processing failed: {e}")

@handler.add(MessageEvent, message=AudioMessageContent)
def handle_voice_message(event: MessageEvent):
    user_id = event.source.user_id
    message_id = event.message.id
    logger.info(f"[VOICE] Voice from {user_id}")
    
    # Use standard threading
    threading.Thread(target=process_voice_message, args=(user_id, message_id)).start()

def process_voice_message(user_id: str, message_id: str):
    try:
        with ApiClient(configuration) as api_client:
            api_blob = MessagingApiBlob(api_client)
            content = api_blob.get_message_content(message_id)
        
        audio_path = f"voice_cache/{message_id}.m4a"
        with open(audio_path, 'wb') as f:
            f.write(content)
        
        text = process_voice_vosk(audio_path)
        logger.info(f"STT Result: {text}")
        
        if text:
            # Re-route the transcribed text through the text handler
            intent, confidence = parse_intent(text, ollama_client)
            user_context = admin_handler.get_user_context(user_id)
            response = handle_intent(intent, user_id, user_context, text)
        else:
            response = "Sorry, I couldn't hear that clearly. Could you repeat?"
        
        with ApiClient(configuration) as api_client:
            api = MessagingApi(api_client)
            api.push_message(PushMessageRequest(to=user_id, messages=[TextMessage(text=response)]))
            
        os.remove(audio_path)
    except Exception as e:
        logger.error(f"Voice processing failed: {e}")

# ============================================================================
# INTENT HANDLERS & RAG PIPELINE
# ============================================================================

@handler.add(MessageEvent, message=FileMessageContent)
def handle_file_message(event: MessageEvent):
    """Catch direct file uploads and tell the user to use Drive instead."""
    user_id = event.source.user_id
    logger.info(f"[FILE] Direct file upload attempted by {user_id}")
    
    msg = "I see you uploaded a file directly! To add documents to my memory, please upload them to your shared Google Drive folder instead, and then type 'Scan drive'."
    
    with ApiClient(configuration) as api_client:
        api = MessagingApi(api_client)
        api.push_message(PushMessageRequest(to=user_id, messages=[TextMessage(text=msg)]))

def handle_intent(intent: str, user_id: str, context: Dict[str, Any], text: str = "") -> str:
    """Route intent to admin commands, drive actions, or RAG querying."""
    
    if intent.startswith("ADMIN_"):
        return admin_handler.execute(intent, user_id, text, context)
        
    elif intent == "DRIVE_SCAN":
        folder_id = context.get('folder_id')
        if not folder_id:
            return "You don't have a linked Drive folder yet. Ask an admin to add you."
        try:
            documents = drive_handler.scan_user_folder(user_id=user_id, folder_id=folder_id)
            if not documents:
                return "Your Drive folder is empty or contains no supported files."
            count = drive_handler.batch_embed_documents(documents=documents, user_id=user_id)
            return f"Drive scan complete!\n\nFound {len(documents)} files. Memorized {count} text chunks."
        except Exception as e:
            return f"Error scanning drive: {str(e)}"
            
    # --- FIX: THE RAG PIPELINE (Retrieval-Augmented Generation) ---
    elif intent in ["INVENTORY_LOOKUP", "REPAIR_SUGGEST", "INVENTORY_UPDATE"]:
        try:
            if not drive_handler.chroma_client:
                return "Database not initialized."
                
            # 1. Look up relevant documents in ChromaDB
            collection = drive_handler.chroma_client.get_collection(name=f"drive_user_{user_id}")
            query_embedding = drive_handler.encoder.encode(text, convert_to_tensor=False).tolist()
            
            results = collection.query(
                query_embeddings=[query_embedding], 
                n_results=3 # Fetch top 3 most relevant paragraphs
            )
            
            if not results['documents'] or not results['documents'][0]:
                return f"I couldn't find any information in your Drive files regarding '{text}'."
                
            # 2. Combine the retrieved paragraphs
            context_text = "\n---\n".join(results['documents'][0])
            
            # 3. Pass the context and the question to Gemini to generate a human-like answer
            from google import genai
            client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
            
            prompt = (
                f"You are a helpful business assistant. Using ONLY the following information retrieved from the user's database, "
                f"answer their query. Be concise and professional.\n\n"
                f"DATABASE DOCUMENTS:\n{context_text}\n\n"
                f"USER QUERY: {text}"
            )
            
            response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return f"Sorry, I ran into an error searching your documents: {e}"
            
    else:
        return f"Intent: {intent} registered. Functionality is still being built!"

def handle_gemini_escalation(text: str, user_id: str) -> str:
    try:
        from google import genai
        client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        prompt = f"User query: {text}\nProvide a concise response."
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini escalation failed: {e}")
        return "Sorry, I couldn't process that. Please try again."

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    uvicorn_run(app, host='0.0.0.0', port=port, log_level='info')