#!/usr/bin/env python3
"""
Local LLM Client: Strict prompt templating to prevent prompt leaking.
Uses Ollama/Typhoon with proper prompt sandwich format.
"""

import os
import logging
import re
from typing import Optional
from collections import defaultdict
import requests
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

def is_thai_text(text: str) -> bool:
    """Check if text contains Thai characters."""
    return bool(re.search(r'[\u0E00-\u0E7F]', text))

class OllamaLLMClient:
    """Strict Ollama LLM client with proper prompt templating."""
    
    def __init__(self, host: str = 'http://127.0.0.1:11434', model: str = 'scb10x/llama3.2-typhoon2-3b-instruct'):
        self.host = host
        self.model = model
        logger.info(f"OllamaLLMClient initialized (model={model})")
    
    def generate(self, prompt: str, context: Optional[str] = "", history: str = "") -> Optional[str]:
        """
        Generate response using strict prompt templating.
        Uses <s>[INST]...[/INST]</s> format to prevent prompt leaking.
        Returns ONLY the generated content, strips all context/history from output.
        """
        # Construct prompt using strict templating format
        instruction = (
            f"You are a helpful factory assistant. Answer the user's question accurately and concisely.\n\n"
        )
        
        if context:
            instruction += f"RELEVANT INFORMATION:\n{context}\n\n"
        
        instruction += f"USER QUERY: {prompt}\n\nProvide ONLY your direct answer. Do not repeat the user's question or context."
        
        # Use proper prompt template format for Typhoon
        full_prompt = f"<s>[INST] {instruction} [/INST]"
        
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
                        "num_predict": 300,
                        "top_k": 40,
                        "top_p": 0.9
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            response_text = response.json().get('response', '').strip()
            
            # CRITICAL: Strip out any leaked prompt/context from the response
            response_text = self._strip_prompt_leakage(response_text, prompt, context, history)
            
            return response_text if response_text else None
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama query failed: {e}")
            return None
    
    @staticmethod
    def _strip_prompt_leakage(response: str, prompt: str, context: str, history: str) -> str:
        """
        Remove echoed prompt, context, and history from response.
        Ensures only the generated answer is returned.
        """
        # Remove common leakage patterns
        patterns = [
            r"RECENT CONVERSATION HISTORY:.*?(?=\n\n|\Z)",  # History echo
            r"DATABASE DOCUMENTS:.*?(?=\n\n|\Z)",  # Context echo
            r"USER QUERY:.*?(?=\n\n|\Z)",  # Query echo
            r"\[INST\].*?\[/INST\]",  # Instruction tags
            r"<s>|</s>",  # Special tokens
            r"You are a helpful.*?assistant\.",  # System prompt
        ]
        
        cleaned = response
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Clean up excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        
        return cleaned if cleaned else response


def query_gemini_fallback(prompt: str, context: str = "", history: str = "") -> Optional[str]:
    """
    Gemini as cloud fallback with cleaner output.
    """
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return None
        
        client = genai.Client(api_key=api_key)
        
        full_prompt = (
            f"You are a helpful factory assistant. Answer accurately and concisely.\n\n"
        )
        
        if context:
            full_prompt += f"RELEVANT INFORMATION:\n{context}\n\n"
        
        full_prompt += f"USER QUERY: {prompt}\n\nProvide ONLY your direct answer."
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=types.GenerateContentConfig(max_output_tokens=300, temperature=0.5)
        )
        return response.text.strip() if response else None
    except Exception as e:
        logger.error(f"Gemini query failed: {e}")
        return None


def generate_smart_response(prompt: str, context: Optional[str] = "", history: str = "", llm_client: Optional[OllamaLLMClient] = None) -> str:
    """
    Intelligent response routing: local LLM first for Thai, fallback to Gemini.
    """
    # 1. Try Local Model first if Thai
    if is_thai_text(prompt) or is_thai_text(context or ""):
        if llm_client:
            response = llm_client.generate(prompt, context, history)
            if response:
                return response
            logger.warning("Local Typhoon inference failed. Falling back to Gemini API.")
    
    # 2. Try Cloud API
    logger.info("Routing to cloud Gemini API...")
    response = query_gemini_fallback(prompt, context, history)
    
    # 3. ULTIMATE FAILSAFE: Force Local Model if Cloud fails
    if not response and llm_client:
        logger.warning("Gemini API failed. Forcing local model takeover.")
        response = llm_client.generate(prompt, context, history)
    
    return response if response else "ขออภัยครับ ระบบประมวลผลคลาวด์และโลคอลขัดข้องชั่วคราว กรุณารอสักครู่"


USER_MESSAGE_HISTORY = defaultdict(list)

def detect_spam(text: str, user_id: str) -> bool:
    """Basic spam detection."""
    if len(text) > 1000:
        return True
    return False
