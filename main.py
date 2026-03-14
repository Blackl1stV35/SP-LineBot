#!/usr/bin/env python3
"""
SP-LineBot: Production-ready Line Bot.
Multimodal (text, image, voice) with RAG and Conversational Memory.
"""

import os
import sys
import threading
from collections import defaultdict, deque
from dotenv import load_dotenv

load_dotenv()

for d in ['logs', 'uploads', 'voice_cache']:
    os.makedirs(d, exist_ok=True)
    
import json
import logging
from datetime import datetime, timezone
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

# Custom modules
from local_llm import OllamaLLMClient, parse_intent, detect_spam, generate_smart_response
from multimodal import process_image_ocr, process_voice_vosk, extract_metadata_and_embed
from drive_handler import DriveHandler
from admin_commands import AdminCommandHandler

# ============================================================================
# CONFIG & MEMORY BUFFER
# ============================================================================

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

# MEMORY: Keeps the last 5 interactions per user
USER_CHAT_HISTORY = defaultdict(lambda: deque(maxlen=5))

app = FastAPI(title="SP-LineBot", version="1.0.0")

LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')

handler = WebhookHandler(LINE_CHANNEL_SECRET)
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)

# GLOBAL INSTANCES (Prevents initializing models on every message)
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
ollama_client = OllamaLLMClient(host=OLLAMA_HOST)

GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON', 'google-service-account.json')
drive_handler = DriveHandler(GOOGLE_SERVICE_ACCOUNT_JSON)
admin_handler = AdminCommandHandler(drive_handler)

@app.on_event('startup')
async def startup_event():
    logger.info("=" * 60)
    logger.info("SP-LineBot Starting Up - Production Mode")
    drive_handler.authenticate()
    logger.info("=" * 60)

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
    
    if detect_spam(text, user_id): return
    
    user_context = admin_handler.get_user_context(user_id)
    intent, confidence = parse_intent(text, ollama_client)
    
    if confidence > 0.7:
        response = handle_intent(intent, user_id, user_context, text)
    else:
        response = handle_gemini_escalation(text, user_id)
        
    with ApiClient(configuration) as api_client:
        api = MessagingApi(api_client)
        api.push_message(PushMessageRequest(to=user_id, messages=[TextMessage(text=response)]))

@handler.add(MessageEvent, message=AudioMessageContent)
def handle_voice_message(event: MessageEvent):
    user_id = event.source.user_id
    message_id = event.message.id
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
            intent, confidence = parse_intent(text, ollama_client)
            user_context = admin_handler.get_user_context(user_id)
            if confidence > 0.7:
                response = handle_intent(intent, user_id, user_context, text)
            else:
                response = handle_gemini_escalation(text, user_id)
        else:
            response = "ขออภัยครับ ฟังไม่ค่อยถนัด รบกวนพูดอีกครั้งได้ไหมครับ"
        
        with ApiClient(configuration) as api_client:
            api = MessagingApi(api_client)
            api.push_message(PushMessageRequest(to=user_id, messages=[TextMessage(text=response)]))
            
        os.remove(audio_path)
    except Exception as e:
        logger.error(f"Voice processing failed: {e}")

def handle_intent(intent: str, user_id: str, context: Dict[str, Any], text: str = "") -> str:
    if intent.startswith("ADMIN_"):
        return admin_handler.execute(intent, user_id, text, context)
        
    elif intent == "DRIVE_SCAN":
        folder_id = context.get('folder_id')
        if not folder_id: return "You don't have a linked Drive folder yet."
        try:
            documents = drive_handler.scan_user_folder(user_id=user_id, folder_id=folder_id)
            count = drive_handler.batch_embed_documents(documents=documents, user_id=user_id)
            return f"อัพเดทฐานข้อมูลเรียบร้อยแล้วครับ! เจอ {len(documents)} ไฟล์ (จำข้อมูลได้ {count} ส่วน)"
        except Exception as e: return f"Error scanning drive: {str(e)}"
            
    elif intent in ["INVENTORY_LOOKUP", "REPAIR_SUGGEST", "INVENTORY_UPDATE"]:
        try:
            collection = drive_handler.chroma_client.get_collection(name=f"drive_user_{user_id}")
            query_embedding = drive_handler.encoder.encode(text, convert_to_tensor=False).tolist()
            
            results = collection.query(query_embeddings=[query_embedding], n_results=3)
            context_text = "\n---\n".join(results['documents'][0]) if results['documents'] else ""
            
            # FORMAT HISTORY
            history = "\n".join([f"User: {h['user']}\nBot: {h['bot']}" for h in USER_CHAT_HISTORY[user_id]])
            
            # FIXED: Added the 'return' keyword and passed the global client
            response_text = generate_smart_response(
                prompt=text, 
                context=context_text, 
                history=history, 
                llm_client=ollama_client
            )
            
            USER_CHAT_HISTORY[user_id].append({"user": text, "bot": response_text})
            return response_text
            
        except Exception as e: return f"Error searching documents: {e}"
            
    return f"Intent: {intent} registered but not processed."

def handle_gemini_escalation(text: str, user_id: str) -> str:
    history = "\n".join([f"User: {h['user']}\nBot: {h['bot']}" for h in USER_CHAT_HISTORY[user_id]])
    response_text = generate_smart_response(prompt=text, context="", history=history, llm_client=ollama_client)
    USER_CHAT_HISTORY[user_id].append({"user": text, "bot": response_text})
    return response_text

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    uvicorn_run(app, host='0.0.0.0', port=port, log_level='info')