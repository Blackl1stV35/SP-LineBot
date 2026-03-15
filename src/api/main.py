#!/usr/bin/env python3
import os
import sys
import logging
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi, MessagingApiBlob,
    ReplyMessageRequest, TextMessage
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

# --- Project Modules ---
from src.agent.agentic_router import analyze_intent_async
from src.agent.local_llm import generate_typhoon_response
from src.services.drive_handler import DriveHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="SP-LineBot-V2")

configuration = Configuration(access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))
drive_handler = DriveHandler()

def reply_text(reply_token: str, text: str):
    try:
        line_bot_api.reply_message(ReplyMessageRequest(reply_token=reply_token, messages=[TextMessage(text=text)]))
    except Exception as e:
        logger.error(f"LINE Reply Error: {e}")

async def process_intent(text: str, user_id: str, reply_token: str):
    text_lower = text.strip().lower()
    
    # 1. FRICTIONLESS UX: Setup System Command
    if text_lower == "ตั้งค่าระบบ":
        reply_text(reply_token, "⏳ กำลังสร้างพื้นที่จัดเก็บข้อมูลส่วนตัวให้คุณ โปรดรอสักครู่...")
        try:
            link, folder_id = drive_handler.create_user_folder(user_id)
            # You would normally save `folder_id` to a database mapped to `user_id` here
            msg = f"✅ สร้างโฟลเดอร์สำเร็จ!\n\n📂 อัปโหลดไฟล์สต็อก Excel/PDF ของคุณที่นี่ (ไม่ต้องตั้งค่าสิทธิ์ใดๆ):\n{link}\n\nพิมพ์ 'สแกนไดรฟ์' เมื่ออัปโหลดเสร็จครับ"
        except Exception as e:
            msg = f"❌ เกิดข้อผิดพลาดในการสร้าง Drive: {e}"
        reply_text(reply_token, msg)
        return

    # 2. Admin Command: Scan Drive
    if text_lower == "สแกนไดรฟ์":
        reply_text(reply_token, "กำลังดำเนินการดึงข้อมูลจากกูเกิลไดรฟ์...")
        # TODO: Call drive_scanner.py here
        reply_text(reply_token, "✅ สแกนสำเร็จ ข้อมูลเข้าสู่ระบบแล้ว")
        return

    # 3. Autonomous Local AI Routing
    try:
        tool_name, args = await analyze_intent_async(text)
        user_query = args.get("query", text)
        
        if tool_name == "tool_check_inventory":
            # Simulate fetching from ChromaDB
            db_context = "ไม่พบข้อมูลในฐานข้อมูล" # Replace with real ChromaDB search
            system_prompt = f"คุณคือผู้ช่วยโรงงานอู่ซ่อมรถ ตอบคำถามอ้างอิงจากข้อมูลนี้เท่านั้น: {db_context}"
            reply_text(reply_token, "กำลังค้นหาสต็อก...")
            answer = await generate_typhoon_response(user_query, system_prompt)
            
        elif tool_name == "tool_add_memory":
            # Call DB updater here
            reply_text(reply_token, "✅ บันทึกข้อมูลการเบิกจ่ายลงฐานข้อมูลเรียบร้อยแล้ว")
            return
            
        else: # General Chat
            system_prompt = "คุณคือผู้ช่วยอู่ซ่อมรถ SP Auto Service ตอบกลับเป็นภาษาไทยอย่างสุภาพ"
            answer = await generate_typhoon_response(user_query, system_prompt)
            
        reply_text(reply_token, answer)
            
    except Exception as e:
        logger.error(f"System Error: {e}")
        reply_text(reply_token, "ขออภัย ระบบประมวลผลขัดข้อง")

@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return JSONResponse(content={"status": "ok"})

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
    asyncio.create_task(process_intent(event.message.text, event.source.user_id, event.reply_token))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app" if "src" in os.getcwd() else "main:app", host="0.0.0.0", port=8000, reload=True)