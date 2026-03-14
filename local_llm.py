#!/usr/bin/env python3
"""
Local LLM Client: Ollama (Typhoon) + Gemini (fallback).
Intent parsing with FIXED_COMMANDS similarity.
Spam detection for abuse prevention.
"""

import os
import json
import logging
import re
from typing import Optional, Tuple, Dict, List
from datetime import datetime, timedelta
from collections import defaultdict

import torch
import requests
from sentence_transformers import SentenceTransformer, util
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ============================================================================
# SMART LANGUAGE DETECTOR
# ============================================================================

def is_thai_text(text: str) -> bool:
    """
    Lightning-fast check to see if the text contains Thai characters.
    Thai Unicode range is \u0E00-\u0E7F.
    """
    if re.search(r'[\u0E00-\u0E7F]', text):
        return True
    return False

# ============================================================================
# FIXED COMMANDS & EMBEDDINGS (UPDATED FOR THAI)
# ============================================================================

FIXED_COMMANDS = {
    "ADMIN_ADD_USER": ["add user", "new member", "register user", "create account"],
    "ADMIN_DEL_USER": ["remove user", "delete member", "kick user"],
    "ADMIN_LIST_USERS": ["list users", "show members", "who is member"],
    "DRIVE_SCAN": ["scan drive", "fetch files", "load documents", "fetch data", "สแกนไดรฟ์", "อัพเดทไฟล์", "ดึงข้อมูล"],
    "INVENTORY_LOOKUP": [
        "check stock", "inventory status", "product info", "how much", 
        "ใครเบิก", "เบิกอะไรไปบ้าง", "เช็คสต็อก", "เหลือเท่าไหร่", "มีของไหม", "ตรวจสอบวัสดุ"
    ],
    "INVENTORY_UPDATE": ["update stock", "add inventory", "reduce quantity", "เพิ่มสต็อก", "ลดสต็อก", "อัพเดทจำนวน"],
    "REPAIR_SUGGEST": ["repair advice", "fix suggestion", "how to repair", "ซ่อมยังไง", "วิธีซ่อม", "แนะนำการซ่อม"],
    "FALLBACK": ["unknown", "help", "info"]
}

class OllamaLLMClient:
    """Ollama local LLM client using SCB 10X Typhoon for Thai."""
    
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
            
    def generate(self, prompt: str, context: Optional[str] = None) -> Optional[str]:
        """Generate response directly from local Typhoon via Ollama."""
        full_prompt = (
            f"You are a helpful factory assistant. Using ONLY the following information retrieved from the user's database, "
            f"answer their query accurately in Thai. Do not make up information.\n\n"
            f"DATABASE DOCUMENTS:\n{context}\n\nUSER QUERY: {prompt}"
        ) if context else prompt
        
        try:
            logger.info("Routing query to local Typhoon model...")
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.5, # Lowered for factual consistency
                        "num_predict": 300
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama/Typhoon query failed: {e}")
            return None

# ============================================================================
# SMART ROUTER (THAI -> TYPHOON, ENGLISH -> GEMINI)
# ============================================================================

def query_gemini_fallback(prompt: str, context: str = "") -> Optional[str]:
    """Query Gemini API directly."""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return None
            
        client = genai.Client(api_key=api_key)
        
        full_prompt = (
            f"You are a helpful factory assistant. Using ONLY the following information retrieved from the user's database, "
            f"answer their query accurately. Do not make up information.\n\n"
            f"DATABASE DOCUMENTS:\n{context}\n\nUSER QUERY: {prompt}"
        ) if context else prompt
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=types.GenerateContentConfig(max_output_tokens=300, temperature=0.5)
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini query failed: {e}")
        return None

def generate_smart_response(prompt: str, context: Optional[str] = "") -> str:
    """Routes query based on language to save tokens and utilize local offline processing."""
    
    # Check if the query contains Thai
    if is_thai_text(prompt) or is_thai_text(context):
        # Local Offline execution
        ollama_client = OllamaLLMClient()
        response = ollama_client.generate(prompt, context)
        
        if response:
            return response
        else:
            logger.warning("Typhoon offline inference failed. Falling back to Gemini API.")
            
    # English query OR Local model failed
    logger.info("Routing to cloud Gemini API...")
    response = query_gemini_fallback(prompt, context)
    return response if response else "System error: I could not process your request at this time."

# ============================================================================
# INTENT PARSING & SPAM
# ============================================================================

def parse_intent(text: str, ollama_client: Optional[OllamaLLMClient] = None) -> Tuple[str, float]:
    if not text:
        return "FALLBACK", 0.0
        
    text_lower = text.lower()
    
    # Shortcuts
    if "add user" in text_lower:
        return "ADMIN_ADD_USER", 1.0
    elif "delete user" in text_lower or "remove user" in text_lower:
        return "ADMIN_DEL_USER", 1.0
    elif "list users" in text_lower:
        return "ADMIN_LIST_USERS", 1.0

    if not ollama_client or not ollama_client.encoder:
        return "FALLBACK", 0.0
        
    try:
        query_embedding = ollama_client.encoder.encode(text_lower, convert_to_tensor=True)
        
        best_intent = "FALLBACK"
        best_confidence = 0.0
        
        for intent, phrases in FIXED_COMMANDS.items():
            if intent.startswith("ADMIN_"): 
                continue
                
            phrase_embeddings = ollama_client.encoder.encode(phrases, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, phrase_embeddings)[0]
            max_score = torch.max(cos_scores).item()
            
            if max_score > best_confidence:
                best_confidence = max_score
                best_intent = intent
                
        logger.info(f"Intent parse: '{text_lower[:50]}' -> {best_intent} (conf={best_confidence:.2f})")
        
        if best_confidence >= 0.7:
            return best_intent, best_confidence
        else:
            return "FALLBACK", best_confidence
            
    except Exception as e:
        logger.error(f"Intent parsing failed: {e}")
        return "FALLBACK", 0.0

USER_MESSAGE_HISTORY = defaultdict(list)
SPAM_THRESHOLD = 10 

def detect_spam(text: str, user_id: str) -> bool:
    now = datetime.utcnow()
    user_history = USER_MESSAGE_HISTORY[user_id]
    user_history[:] = [(msg, ts) for msg, ts in user_history if (now - ts).total_seconds() < 60]
    user_history.append((text, now))
    
    recent_10s = [msg for msg, ts in user_history if (now - ts).total_seconds() < 10]
    if len(recent_10s) >= 3 and len(set(recent_10s)) == 1:
        return True
    
    recent_1m = [msg for msg, ts in user_history if (now - ts).total_seconds() < 60]
    if len(recent_1m) > SPAM_THRESHOLD:
        return True
    
    if len(text) > 1000:
        return True
        
    return False