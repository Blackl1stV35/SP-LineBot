#!/usr/bin/env python3
"""
Agentic Router: Intent classification with async support and extended timeouts.
Routes user messages to appropriate tools: scan_drive, check_inventory, add_memory, general_chat.
"""

import os
import json
import logging
import asyncio
from typing import Tuple, Dict, Any, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
import requests
import httpx

load_dotenv()
logger = logging.getLogger(__name__)

# Extended timeout for GPU-heavy operations
OLLAMA_TIMEOUT = 120  # 2 minutes for local inference
GEMINI_TIMEOUT = 30   # 30 seconds for cloud API

def tool_scan_drive():
    """Tool to scan user's Google Drive and update the vector database."""
    pass

def tool_check_inventory(search_query: str):
    """Tool to search inventory records and check stock levels."""
    pass

def tool_add_memory(note: str):
    """Tool to record new data, update stock, or save notes."""
    pass

def tool_general_chat():
    """Tool for general conversation, greetings, or non-inventory questions."""
    pass


def fallback_analyze_intent_sync(user_message: str, history: str = "") -> Tuple[str, Dict[str, Any]]:
    """
    Synchronous local Ollama fallback for intent classification.
    Returns strict JSON routing with extended timeouts for GPU inference.
    """
    logger.info("Triggering Local Ollama Fallback for Agentic Router...")
    
    prompt = f"""You are an intent classification system for a factory inventory bot.
    Choose the BEST tool for the user's message.
    
    TOOLS:
    1. "tool_scan_drive" : Use when the user wants to scan drive or update files. (No args)
    2. "tool_check_inventory" : Use to check stock or see who drew items. Args: "search_query" (what to search).
    3. "tool_add_memory" : Use to record new data or save notes. Args: "note" (the summary to save).
    4. "tool_general_chat" : Use for greetings, general questions, or anything else. (No args)
    
    RECENT CONVERSATION HISTORY (Use this for context):
    {history}
    
    CURRENT USER MESSAGE: "{user_message}"
    
    Respond STRICTLY with ONLY a JSON object, nothing else:
    {{"tool": "tool_name", "args": {{"arg_key": "arg_value"}}}}
    """
    
    host = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
    model = 'scb10x/llama3.2-typhoon2-3b-instruct'
    
    try:
        response = requests.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0}
            },
            timeout=OLLAMA_TIMEOUT  # Extended timeout for GPU inference
        )
        response.raise_for_status()
        result_text = response.json().get('response', '').strip()
        
        parsed = json.loads(result_text)
        tool_name = parsed.get("tool", "tool_general_chat")
        args_dict = parsed.get("args", {})
        
        valid_tools = ["tool_scan_drive", "tool_check_inventory", "tool_add_memory", "tool_general_chat"]
        if tool_name not in valid_tools:
            tool_name = "tool_general_chat"
        
        logger.info(f"Local Router Decision: {tool_name} {args_dict}")
        return tool_name, args_dict
        
    except Exception as e:
        logger.error(f"Ollama Fallback Router failed: {e}")
        return "tool_general_chat", {}


async def fallback_analyze_intent_async(user_message: str, history: str = "") -> Tuple[str, Dict[str, Any]]:
    """
    Asynchronous local Ollama fallback using httpx.
    Allows FastAPI to remain responsive during GPU inference.
    """
    logger.info("Triggering Async Local Ollama Fallback...")
    
    prompt = f"""You are an intent classification system for a factory inventory bot.
    Choose the BEST tool for the user's message.
    
    TOOLS:
    1. "tool_scan_drive" : Use when the user wants to scan drive or update files. (No args)
    2. "tool_check_inventory" : Use to check stock or see who drew items. Args: "search_query" (what to search).
    3. "tool_add_memory" : Use to record new data or save notes. Args: "note" (the summary to save).
    4. "tool_general_chat" : Use for greetings, general questions, or anything else. (No args)
    
    RECENT CONVERSATION HISTORY:
    {history}
    
    CURRENT USER MESSAGE: "{user_message}"
    
    Respond STRICTLY with ONLY a JSON object:
    {{"tool": "tool_name", "args": {{"arg_key": "arg_value"}}}}
    """
    
    host = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
    model = 'scb10x/llama3.2-typhoon2-3b-instruct'
    
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(
                f"{host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.0}
                }
            )
            response.raise_for_status()
            result_text = response.json().get('response', '').strip()
            
            parsed = json.loads(result_text)
            tool_name = parsed.get("tool", "tool_general_chat")
            args_dict = parsed.get("args", {})
            
            valid_tools = ["tool_scan_drive", "tool_check_inventory", "tool_add_memory", "tool_general_chat"]
            if tool_name not in valid_tools:
                tool_name = "tool_general_chat"
            
            logger.info(f"Async Local Router Decision: {tool_name}")
            return tool_name, args_dict
    except Exception as e:
        logger.error(f"Async Ollama Fallback failed: {e}")
        return "tool_general_chat", {}


def analyze_intent(user_message: str, history: str = "") -> Tuple[str, Dict[str, Any]]:
    """
    Primary Cloud Brain: Evaluates user message using Gemini API.
    Falls back to local Ollama on API failure.
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    try:
        contextual_prompt = f"Recent History:\n{history}\n\nAnalyze this request and call the right tool: {user_message}"
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contextual_prompt,
            config=types.GenerateContentConfig(
                tools=[tool_scan_drive, tool_check_inventory, tool_add_memory, tool_general_chat],
                temperature=0.0
            )
        )
        
        if response.function_calls:
            call = response.function_calls[0]
            args_dict = {k: v for k, v in call.args.items()} if call.args else {}
            logger.info(f"Cloud Router Decision: {call.name}")
            return call.name, args_dict
        
        return "tool_general_chat", {}
    except Exception as e:
        logger.warning(f"Gemini Router Error (falling back to local): {e}")
        return fallback_analyze_intent_sync(user_message, history)


async def analyze_intent_async(user_message: str, history: str = "") -> Tuple[str, Dict[str, Any]]:
    """
    Async wrapper for intent analysis. Falls back to local if Gemini fails.
    Designed for use in FastAPI endpoints to avoid blocking the event loop.
    """
    try:
        # Run the sync Gemini call in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, analyze_intent, user_message, history)
    except Exception as e:
        logger.warning(f"Async intent analysis failed, using local fallback: {e}")
        return await fallback_analyze_intent_async(user_message, history)
