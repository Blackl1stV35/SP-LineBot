#!/usr/bin/env python3
"""
Local LLM Client: Ollama (primary) + Gemini (fallback).
Intent parsing with FIXED_COMMANDS similarity (conf > 0.7).
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
# UPDATED: Import the new GenAI SDK and types
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ============================================================================
# FIXED COMMANDS & EMBEDDINGS
# ============================================================================

FIXED_COMMANDS = {
    "ADMIN_ADD_USER": ["add user", "new member", "register user", "create account"],
    "ADMIN_DEL_USER": ["remove user", "delete member", "kick user"],
    "ADMIN_LIST_USERS": ["list users", "show members", "who is member"],
    "DRIVE_SCAN": ["scan drive", "fetch files", "load documents", "fetch data"],
    "INVENTORY_LOOKUP": ["check stock", "inventory status", "product info", "how much"],
    "INVENTORY_UPDATE": ["update stock", "add inventory", "reduce quantity"],
    "REPAIR_SUGGEST": ["repair advice", "fix suggestion", "how to repair"],
    "FALLBACK": ["unknown", "help", "info"]
}

class OllamaLLMClient:
    """Ollama local LLM client with batch inference."""
    
    def __init__(self, host: str = 'http://127.0.0.1:11434', model: str = 'mistral'):
        self.host = host
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"OllamaLLMClient init: model={model}, device={self.device}")
        
        # Load sentence transformer for semantic similarity
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer loaded (semantic similarity)")
        except Exception as e:
            logger.error(f"Sentence transformer load failed: {e}")
            self.encoder = None
    
    async def health_check(self) -> Dict:
        """Check Ollama server health."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            return {"status": "ok", "models": len(data.get('models', []))}
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 256) -> str:
        """Generate response from Ollama local LLM."""
        try:
            logger.debug(f"Ollama generate: {prompt[:80]}...")
            
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            text = result.get('response', '').strip()
            logger.debug(f"Ollama response: {text[:100]}...")
            return text
        except requests.exceptions.Timeout:
            logger.error("Ollama timeout")
            return ""
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return ""
    
    async def embed(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings for texts (batch)."""
        if not self.encoder:
            logger.warning("Sentence transformer not available")
            return None
        
        try:
            embeddings = self.encoder.encode(texts, convert_to_tensor=True, device=self.device)
            return embeddings
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

# ============================================================================
# INTENT PARSING
# ============================================================================

def parse_intent(text: str, ollama_client: Optional[OllamaLLMClient] = None) -> Tuple[str, float]:
    """Parse user intent using hardcoded rules first, then semantic similarity."""
    if not text:
        return "FALLBACK", 0.0
        
    text_lower = text.lower()
    
    # --- FIX 2: Hardcoded Shortcuts for Admin Commands ---
    if "add user" in text_lower:
        logger.info(f"Intent parse shortcut: '{text_lower[:50]}' -> ADMIN_ADD_USER (conf=1.00)")
        return "ADMIN_ADD_USER", 1.0
    elif "delete user" in text_lower or "remove user" in text_lower:
        logger.info(f"Intent parse shortcut: '{text_lower[:50]}' -> ADMIN_DEL_USER (conf=1.00)")
        return "ADMIN_DEL_USER", 1.0
    elif "list users" in text_lower:
        logger.info(f"Intent parse shortcut: '{text_lower[:50]}' -> ADMIN_LIST_USERS (conf=1.00)")
        return "ADMIN_LIST_USERS", 1.0

    # --- Standard Semantic Similarity for other commands ---
    if not ollama_client or not ollama_client.encoder:
        return "FALLBACK", 0.0
        
    try:
        query_embedding = ollama_client.encoder.encode(text_lower, convert_to_tensor=True)
        
        best_intent = "FALLBACK"
        best_confidence = 0.0
        
        for intent, phrases in FIXED_COMMANDS.items():
            if intent.startswith("ADMIN_"): # Skip admin commands since we handled them above
                continue
                
            phrase_embeddings = ollama_client.encoder.encode(phrases, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, phrase_embeddings)[0]
            max_score = torch.max(cos_scores).item()
            
            if max_score > best_confidence:
                best_confidence = max_score
                best_intent = intent
                
        # FIX 1: Replaced the fancy '->' with a standard '->' to prevent Windows Unicode crashes
        logger.info(f"Intent parse: '{text_lower[:50]}' -> {best_intent} (conf={best_confidence:.2f})")
        
        if best_confidence >= 0.7:
            return best_intent, best_confidence
        else:
            return "FALLBACK", best_confidence
            
    except Exception as e:
        logger.error(f"Intent parsing failed: {e}")
        return "FALLBACK", 0.0

# ============================================================================
# SPAM DETECTION
# ============================================================================

USER_MESSAGE_HISTORY = defaultdict(list)
SPAM_THRESHOLD = 10  # Messages per minute
BAN_DURATION = 300  # 5 minutes

def detect_spam(text: str, user_id: str) -> bool:
    """
    Detect spam:
    - Repeated identical messages
    - Message flood (>10 msgs/min)
    - Excessive length (>1000 chars)
    """
    now = datetime.utcnow()
    
    # Check message history for user
    user_history = USER_MESSAGE_HISTORY[user_id]
    
    # Clean old messages (>60 seconds)
    user_history[:] = [
        (msg, ts) for msg, ts in user_history
        if (now - ts).total_seconds() < 60
    ]
    
    # Add current message
    user_history.append((text, now))
    
    # Rule 1: Identical message spam (3+ in 10 seconds)
    recent_10s = [msg for msg, ts in user_history if (now - ts).total_seconds() < 10]
    if len(recent_10s) >= 3 and len(set(recent_10s)) == 1:
        logger.warning(f"Spam detected (repeat): {user_id}")
        return True
    
    # Rule 2: Message flood (>10 msgs/min)
    recent_1m = [msg for msg, ts in user_history if (now - ts).total_seconds() < 60]
    if len(recent_1m) > SPAM_THRESHOLD:
        logger.warning(f"Spam detected (flood): {user_id} ({len(recent_1m)} msgs/min)")
        return True
    
    # Rule 3: Excessive length
    if len(text) > 1000:
        logger.warning(f"Spam detected (length): {user_id}")
        return True
    
    # Rule 4: URL/malware patterns
    if re.search(r'(http|ftp)s?://', text) or re.search(r'<script|javascript:/i', text):
        logger.warning(f"Spam detected (URL/malware): {user_id}")
        return True
    
    return False

# ============================================================================
# GEMINI FALLBACK (ASYNC)
# ============================================================================

async def query_gemini_async(prompt: str, context: str = "") -> str:
    """
    Query Gemini API (fallback when Ollama fails).
    Free tier: 1M context, 15 RPM, 1500 RPD.
    """
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.warning("GEMINI_API_KEY not set")
            return None
        
        # UPDATED: Initialize the new Client
        client = genai.Client(api_key=api_key)
        
        full_prompt = f"{context}\n\nQuery: {prompt}" if context else prompt
        
        logger.debug(f"Gemini query: {prompt[:80]}...")
        
        # UPDATED: Use the new generation syntax and types.GenerateContentConfig
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=256,
                temperature=0.7,
            )
        )
        
        result = response.text.strip()
        logger.info(f"Gemini response: {result[:100]}...")
        return result
    except Exception as e:
        logger.error(f"Gemini query failed: {e}")
        return None

# ============================================================================
# BATCH INFERENCE
# ============================================================================

async def batch_intent_parse(texts: List[str], ollama_client: OllamaLLMClient) -> List[Tuple[str, float]]:
    """Batch parse intents for multiple texts (scalability for 40 users)."""
    results = []
    for text in texts:
        intent, conf = parse_intent(text, ollama_client)
        results.append((intent, conf))
    return results