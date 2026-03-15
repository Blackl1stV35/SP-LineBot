import httpx
import logging
import json

logger = logging.getLogger(__name__)
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "typhoon"

async def generate_typhoon_response(prompt: str, system_prompt: str = "") -> str:
    """Calls local Typhoon model to generate a Thai response."""
    
    full_prompt = f"<s>[INST] {system_prompt}\n\nคำถาม: {prompt} [/INST]"
    
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.3, # Low temp for factual factory data
            "num_predict": 250
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "ขออภัย ประมวลผลข้อความไม่สำเร็จ").strip()
    except Exception as e:
        logger.error(f"Typhoon Inference Error: {e}")
        return "⚠️ ระบบ AI ออฟไลน์ขัดข้อง โปรดตรวจสอบ Ollama"