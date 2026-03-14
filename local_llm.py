import os
import logging
import re
from typing import Optional, Tuple, Dict, List
from datetime import datetime
from collections import defaultdict
import torch
import requests
from sentence_transformers import SentenceTransformer, util
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

def is_thai_text(text: str) -> bool:
    return bool(re.search(r'[\u0E00-\u0E7F]', text))

FIXED_COMMANDS = {
    "ADMIN_ADD_USER": ["add user", "new member", "register user", "create account"],
    "ADMIN_DEL_USER": ["remove user", "delete member", "kick user"],
    "ADMIN_LIST_USERS": ["list users", "show members", "who is member"],
    "DRIVE_SCAN": ["scan drive", "fetch files", "load documents", "fetch data", "สแกนไดรฟ์", "อัพเดทไฟล์", "ดึงข้อมูล"],
    "INVENTORY_LOOKUP": ["check stock", "inventory status", "product info", "how much", "ใครเบิก", "เบิกอะไรไปบ้าง", "เบิกอะไรบ้าง", "ใครเบิกบ้าง", "เช็คสต็อก", "เหลือเท่าไหร่", "มีของไหม", "ตรวจสอบวัสดุ", "ข้อมูลการเบิก"],
    "INVENTORY_UPDATE": ["update stock", "add inventory", "reduce quantity", "เพิ่มสต็อก", "ลดสต็อก", "อัพเดทจำนวน"],
    "REPAIR_SUGGEST": ["repair advice", "fix suggestion", "how to repair", "ซ่อมยังไง", "วิธีซ่อม", "แนะนำการซ่อม"],
    "FALLBACK": ["unknown", "help", "info"]
}

class OllamaLLMClient:
    def __init__(self, host: str = 'http://127.0.0.1:11434', model: str = 'scb10x/llama3.2-typhoon2-3b-instruct'):
        self.host = host
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"OllamaLLMClient init: model={model}, device={self.device}")
        
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Sentence transformer load failed: {e}")
            self.encoder = None
            
    def generate(self, prompt: str, context: Optional[str] = "", history: str = "") -> Optional[str]:
        full_prompt = (
            f"You are a helpful factory assistant. Answer the user accurately in Thai.\n\n"
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
                        "temperature": 0.3, # Highly factual
                        "num_predict": 300
                    }
                },
                timeout=120 # Increased timeout to prevent crashes
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
    # Use the existing client to stop HTTP spam
    if is_thai_text(prompt) or is_thai_text(context):
        if llm_client:
            response = llm_client.generate(prompt, context, history)
            if response: return response
            logger.warning("Typhoon offline inference failed. Falling back to Gemini API.")
            
    logger.info("Routing to cloud Gemini API...")
    response = query_gemini_fallback(prompt, context, history)
    return response if response else "System error: I could not process your request at this time."

def parse_intent(text: str, ollama_client: Optional[OllamaLLMClient] = None) -> Tuple[str, float]:
    if not text: return "FALLBACK", 0.0
    text_lower = text.lower()
    
    if "add user" in text_lower: return "ADMIN_ADD_USER", 1.0
    if "delete user" in text_lower or "remove user" in text_lower: return "ADMIN_DEL_USER", 1.0
    if "list users" in text_lower: return "ADMIN_LIST_USERS", 1.0

    if not ollama_client or not ollama_client.encoder: return "FALLBACK", 0.0
        
    try:
        query_embedding = ollama_client.encoder.encode(text_lower, convert_to_tensor=True)
        best_intent, best_confidence = "FALLBACK", 0.0
        
        for intent, phrases in FIXED_COMMANDS.items():
            if intent.startswith("ADMIN_"): continue
            phrase_embeddings = ollama_client.encoder.encode(phrases, convert_to_tensor=True)
            max_score = torch.max(util.cos_sim(query_embedding, phrase_embeddings)[0]).item()
            if max_score > best_confidence:
                best_confidence, best_intent = max_score, intent
                
        logger.info(f"Intent parse: '{text_lower[:50]}' -> {best_intent} (conf={best_confidence:.2f})")
        return (best_intent, best_confidence) if best_confidence >= 0.7 else ("FALLBACK", best_confidence)
    except Exception as e:
        logger.error(f"Intent parsing failed: {e}")
        return "FALLBACK", 0.0

USER_MESSAGE_HISTORY = defaultdict(list)
def detect_spam(text: str, user_id: str) -> bool:
    # Simplified spam logic for brevity in production
    if len(text) > 1000: return True
    return False