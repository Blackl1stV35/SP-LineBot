import os
import json
import requests
import logging
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

def tool_scan_drive():
    """ใช้ฟังก์ชันนี้เมื่อผู้ใช้ต้องการ สแกนไดรฟ์, อัพเดทไฟล์, หรือดึงข้อมูลจาก Google Drive"""
    pass

def tool_check_inventory(search_query: str):
    """ใช้ฟังก์ชันนี้เมื่อผู้ใช้ต้องการ ค้นหาข้อมูลการเบิก, เช็คสต็อก, หรือถามว่าใครเบิกอะไรไปบ้าง
    Args:
        search_query: สิ่งที่ผู้ใช้ต้องการค้นหา เช่น 'สีดำเงา', 'โจ้เบิกอะไร', 'เดือน 2'
    """
    pass

def tool_add_memory(note: str):
    """ใช้ฟังก์ชันนี้เมื่อผู้ใช้ต้องการ บันทึกข้อมูลใหม่, เพิ่มสต็อก, ลดสต็อก, หรือจดบันทึก
    Args:
        note: สรุปข้อความที่ต้องการบันทึก เช่น 'โจ้เบิกสี 2 ป๋อง', 'รับของเข้า 5 ลัง'
    """
    pass

def tool_general_chat():
    """ใช้ฟังก์ชันนี้สำหรับการพูดคุยทั่วไป ทักทาย ถามวิธีซ่อมแซม หรือคำถามที่ไม่เกี่ยวกับสต็อก"""
    pass

def fallback_analyze_intent(user_message: str):
    """
    The Local Brain. Takes over if Google API fails or rate limits.
    Forces Ollama to output a strict JSON routing object.
    """
    logger.info("Triggering Local Ollama Fallback for Agentic Router...")
    
    prompt = f"""You are an intent classification system for a factory inventory bot.
    Choose the best tool for the user's message.
    
    TOOLS:
    1. "tool_scan_drive" : Use when the user wants to scan drive, update files. (No args)
    2. "tool_check_inventory" : Use to check stock or see who drew items. Args: "search_query" (what to search).
    3. "tool_add_memory" : Use to record new data or save notes. Args: "note" (the summary to save).
    4. "tool_general_chat" : Use for greetings, general questions, or anything else. (No args)
    
    USER MESSAGE: "{user_message}"
    
    Respond STRICTLY with ONLY a JSON object in this exact format, nothing else:
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
                "format": "json", # Forces the local LLM to lock into JSON mode
                "options": {"temperature": 0.0} # Absolute zero creativity for routing
            },
            timeout=30
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

def analyze_intent(user_message: str):
    """
    Primary Cloud Brain. Evaluates user message and decides the action.
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"Analyze this request and call the right tool: {user_message}",
            config=types.GenerateContentConfig(
                tools=[tool_scan_drive, tool_check_inventory, tool_add_memory, tool_general_chat],
                temperature=0.0
            )
        )
        
        if response.function_calls:
            call = response.function_calls[0]
            args_dict = {k: v for k, v in call.args.items()} if call.args else {}
            return call.name, args_dict
            
        return "tool_general_chat", {}
    except Exception as e:
        logger.warning(f"Gemini Router Error (Likely Rate Limit/429): {e}")
        # Hand off to the local model instantly
        return fallback_analyze_intent(user_message)

if __name__ == "__main__":
    print("Test Fallback directly:", fallback_analyze_intent("ดารารัตน์เบิกสีดำเงาไปกี่กระป๋อง"))