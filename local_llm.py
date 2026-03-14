import os
import logging
import re
from typing import Optional
from datetime import datetime
from collections import defaultdict
import requests
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

def is_thai_text(text: str) -> bool:
    return bool(re.search(r'[\u0E00-\u0E7F]', text))

class OllamaLLMClient:
    def __init__(self, host: str = 'http://127.0.0.1:11434', model: str = 'scb10x/llama3.2-typhoon2-3b-instruct'):
        self.host = host
        self.model = model
        logger.info(f"OllamaLLMClient init: model={model}")
            
    def generate(self, prompt: str, context: Optional[str] = "", history: str = "") -> Optional[str]:
        full_prompt = (
            f"You are a helpful factory assistant. Answer the user accurately.\n\n"
            f"RECENT CONVERSATION HISTORY:\n{history}\n\n"
            f"DATABASE DOCUMENTS:\n{context}\n\nUSER QUERY: {prompt}"
        )
        
        try:
            logger.info("Routing query to local Typhoon model...")
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3, 
                        "num_predict": 300
                    }
                },
                timeout=120 
            )
            response.raise_for_status()
            return response.json().get('response', '').strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama/Typhoon query failed: {e}")
            return None

def query_gemini_fallback(prompt: str, context: str = "", history: str = "") -> Optional[str]:
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key: return None
        client = genai.Client(api_key=api_key)
        
        full_prompt = (
            f"You are a helpful factory assistant. Answer accurately.\n\n"
            f"RECENT CONVERSATION HISTORY:\n{history}\n\n"
            f"DATABASE DOCUMENTS:\n{context}\n\nUSER QUERY: {prompt}"
        )
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=types.GenerateContentConfig(max_output_tokens=300, temperature=0.5)
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini query failed: {e}")
        return None

def generate_smart_response(prompt: str, context: Optional[str] = "", history: str = "", llm_client: Optional[OllamaLLMClient] = None) -> str:
    # 1. Try Local Model first if Thai
    if is_thai_text(prompt) or is_thai_text(context):
        if llm_client:
            response = llm_client.generate(prompt, context, history)
            if response: return response
            logger.warning("Typhoon offline inference failed. Falling back to Gemini API.")
            
    # 2. Try Cloud API
    logger.info("Routing to cloud Gemini API...")
    response = query_gemini_fallback(prompt, context, history)
    
    # 3. ULTIMATE FAILSAFE: If Cloud API gets 429'd, force Local Model even if English
    if not response and llm_client:
        logger.warning("Gemini API rejected request (Quota/Network). Forcing local model takeover.")
        response = llm_client.generate(prompt, context, history)
        
    return response if response else "ขออภัยครับ ระบบประมวลผลคลาวด์และโลคอลขัดข้องชั่วคราว กรุณารอสักครู่ (System Overload)"

USER_MESSAGE_HISTORY = defaultdict(list)
def detect_spam(text: str, user_id: str) -> bool:
    if len(text) > 1000: return True
    return False