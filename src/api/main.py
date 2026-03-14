#!/usr/bin/env python3
"""
SP-LineBot: Production-ready Agentic LINE Bot (v1.1.0).
Multimodal with RAG, Memory, and DB Write-Back.
Optimized for async, extended timeouts, and hardware utilization.
"""

import os
import sys
import threading
import asyncio
from collections import defaultdict, deque
from dotenv import load_dotenv

load_dotenv()

for d in ['logs', 'uploads', 'voice_cache']:
    os.makedirs(d, exist_ok=True)

import logging
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from uvicorn import run as uvicorn_run
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, MessagingApiBlob, PushMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent, AudioMessageContent

# Import from refactored modules
from src.agent.local_llm import OllamaLLMClient, detect_spam, generate_smart_response
from src.services.multimodal import process_voice_vosk
from src.services.drive_handler import DriveHandler
from src.api.admin_commands import AdminCommandHandler
from src.agent.agentic_router import analyze_intent, analyze_intent_async
from src.db.database import get_db_client, get_embedder_client

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

USER_CHAT_HISTORY = defaultdict(lambda: deque(maxlen=5))

app = FastAPI(title="SP-LineBot", version="1.1.0-Modular")

LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')

handler = WebhookHandler(LINE_CHANNEL_SECRET)
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)

OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
ollama_client = OllamaLLMClient(host=OLLAMA_HOST)
drive_handler = DriveHandler(os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON', 'google-service-account.json'))
admin_handler = AdminCommandHandler(drive_handler)

# Initialize singleton DB clients on startup
db_client = get_db_client()
embedder_client = get_embedder_client()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.1.0"}

@app.post("/callback")
async def webhook(request: Request):
    """LINE webhook callback endpoint."""
    signature = request.headers.get('X-Line-Signature')
    body = await request.body()
    try:
        handler.handle(body.decode('utf-8'), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return JSONResponse({"status": "ok"})

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):
    """Handle incoming text messages."""
    user_id = event.source.user_id
    text = event.message.text
    
    if detect_spam(text, user_id):
        return
    
    user_context = admin_handler.get_user_context(user_id)
    
    # Check for admin commands
    if text.split()[0].isdigit() and "list users" in text.lower():
        response = admin_handler.execute("ADMIN_LIST_USERS", user_id, text, user_context)
    else:
        # Agentic routing with memory
        history_str = "\n".join([f"User: {h['user']}\nBot: {h['bot']}" for h in USER_CHAT_HISTORY[user_id]])
        action, args = analyze_intent(text, history=history_str)
        response = execute_agent_action(action, args, user_id, user_context, text)
    
    with ApiClient(configuration) as api_client:
        api = MessagingApi(api_client)
        api.push_message(PushMessageRequest(to=user_id, messages=[TextMessage(text=response)]))

@handler.add(MessageEvent, message=AudioMessageContent)
def handle_voice_message(event: MessageEvent):
    """Handle incoming voice messages."""
    user_id = event.source.user_id
    message_id = event.message.id
    threading.Thread(target=process_voice_message, args=(user_id, message_id)).start()

def process_voice_message(user_id: str, message_id: str):
    """Process voice message in background thread."""
    try:
        with ApiClient(configuration) as api_client:
            content = MessagingApiBlob(api_client).get_message_content(message_id)
        
        audio_path = f"voice_cache/{message_id}.m4a"
        with open(audio_path, 'wb') as f:
            f.write(content)
        
        text = process_voice_vosk(audio_path)
        logger.info(f"STT Result: {text}")
        
        if text:
            user_context = admin_handler.get_user_context(user_id)
            
            history_str = "\n".join([f"User: {h['user']}\nBot: {h['bot']}" for h in USER_CHAT_HISTORY[user_id]])
            action, args = analyze_intent(text, history=history_str)
            response = execute_agent_action(action, args, user_id, user_context, text)
        else:
            response = "ขออภัยครับ ฟังไม่ถนัด รบกวนพิมพ์หรือพูดอีกครั้งครับ"
        
        with ApiClient(configuration) as api_client:
            MessagingApi(api_client).push_message(PushMessageRequest(to=user_id, messages=[TextMessage(text=response)]))
        os.remove(audio_path)
    except Exception as e:
        logger.error(f"Voice error: {e}")

def execute_agent_action(action: str, args: dict, user_id: str, context: Dict[str, Any], raw_text: str) -> str:
    """Execute agent action based on routing decision."""
    logger.info(f"Agent executing: {action} with args: {args}")
    
    history_str = "\n".join([f"User: {h['user']}\nBot: {h['bot']}" for h in USER_CHAT_HISTORY[user_id]])
    
    if action == "tool_scan_drive":
        folder_id = context.get('folder_id')
        if not folder_id:
            return "คุณยังไม่มีโฟลเดอร์ Drive ที่เชื่อมโยงอยู่ครับ"
        try:
            docs = drive_handler.scan_user_folder(user_id, folder_id)
            count = drive_handler.batch_embed_documents(docs, user_id)
            return f"อัพเดทฐานข้อมูลเรียบร้อยแล้วครับ! ดึงข้อมูลสำเร็จ {count} รายการ"
        except Exception as e:
            return f"Error scanning drive: {e}"
    
    elif action == "tool_check_inventory":
        search_term = args.get('search_query', raw_text)
        try:
            collection = db_client.get_collection(name=f"drive_user_{user_id}")
            if not collection:
                return "กำลังสร้างระบบความจำใหม่ กรุณาพิมพ์ 'สแกนไดรฟ์' ก่อนครับ"
            
            query_emb = embedder_client.encode(search_term, convert_to_tensor=False)
            if query_emb is None:
                return "ระบบการค้นหามีปัญหา โปรดลองใหม่"
            
            results = collection.query(query_embeddings=[query_emb.tolist()], n_results=4)
            context_text = "\n---\n".join(results['documents'][0]) if results['documents'] else ""
            
            response = generate_smart_response(raw_text, context_text, history_str, ollama_client)
            USER_CHAT_HISTORY[user_id].append({"user": raw_text, "bot": response})
            return response
        except Exception as e:
            logger.error(f"Inventory check failed: {e}")
            return "กำลังสร้างระบบความจำใหม่ กรุณาพิมพ์ 'สแกนไดรฟ์' ก่อนครับ"
    
    elif action == "tool_add_memory":
        note = args.get('note', raw_text)
        try:
            collection = db_client.get_or_create_collection(name=f"drive_user_{user_id}")
            
            # Embed the note
            embedding = embedder_client.encode(note, convert_to_tensor=False)
            if embedding is None:
                return "ระบบบันทึกความจำมีปัญหา (Encoder failed)"
            
            doc_id = f"manual_entry_{int(__import__('time').time())}"
            formatted_info = f"ข้อมูลเดือน: บันทึกใหม่\nชื่อพนักงาน: ระบบ/ผู้ใช้\nรายการเบิกวัสดุสิ้นเปลือง: {note}"
            
            collection.add(
                ids=[doc_id],
                embeddings=[embedding.tolist()],
                metadatas=[{'user_id': user_id, 'source': 'manual_update', 'type': 'user_note'}],
                documents=[formatted_info]
            )
            response = f"บันทึกข้อมูลลงในความจำเรียบร้อยแล้วครับ: '{note}'"
            USER_CHAT_HISTORY[user_id].append({"user": raw_text, "bot": response})
            return response
        except Exception as e:
            logger.error(f"DB Write Error: {e}")
            return f"เกิดข้อผิดพลาดในการบันทึก: {str(e)}"
    
    else:  # tool_general_chat
        response = generate_smart_response(raw_text, "", history_str, ollama_client)
        USER_CHAT_HISTORY[user_id].append({"user": raw_text, "bot": response})
        return response

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    uvicorn_run(app, host='0.0.0.0', port=port, log_level='info')
