import logging
import asyncio
from typing import Tuple, Dict, Any
from sentence_transformers import SentenceTransformer, util
import torch

logger = logging.getLogger(__name__)

class SemanticRouter:
    def __init__(self):
        # We use a tiny multilingual model that shares VRAM efficiently
        logger.info("Loading Semantic Router Model...")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define Intents and their semantic trigger phrases
        self.intents = {
            "tool_check_inventory": ["ใครเบิก", "เหลือเท่าไหร่", "สต็อก", "มีของไหม", "เช็คยาง", "เช็คสต็อก"],
            "tool_add_memory": ["จดบันทึก", "เพิ่มข้อมูล", "วันนี้เบิก", "เอาไปใช้", "นำไปใช้", "เบิกของ"],
            "tool_general_chat": ["สวัสดี", "ทำอะไรได้บ้าง", "ขอบคุณ", "ดีจ้า", "สอบถามหน่อย"]
        }
        
        # Pre-compute embeddings for fast matching
        self.intent_embeddings = {}
        for intent, phrases in self.intents.items():
            self.intent_embeddings[intent] = self.model.encode(phrases, convert_to_tensor=True)

    def route_sync(self, text: str) -> Tuple[str, Dict[str, Any]]:
        text_emb = self.model.encode(text, convert_to_tensor=True)
        
        best_intent = "tool_general_chat"
        highest_score = 0.0
        
        for intent, emb_matrix in self.intent_embeddings.items():
            scores = util.cos_sim(text_emb, emb_matrix)[0]
            max_score = torch.max(scores).item()
            
            if max_score > highest_score:
                highest_score = max_score
                best_intent = intent

        # Confidence threshold (if it doesn't match anything well, it's just chat)
        if highest_score < 0.60:
            best_intent = "tool_general_chat"

        logger.info(f"Semantic Router matched {best_intent} (Score: {highest_score:.2f})")
        
        # We pass the raw text as the argument so Typhoon/ChromaDB can parse it later
        return best_intent, {"query": text}

# Singleton instance
_router = SemanticRouter()

async def analyze_intent_async(user_message: str) -> Tuple[str, Dict[str, Any]]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _router.route_sync, user_message)