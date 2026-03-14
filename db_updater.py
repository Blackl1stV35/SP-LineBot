import chromadb
from sentence_transformers import SentenceTransformer
import time
import logging

logger = logging.getLogger(__name__)

class DatabaseUpdater:
    def __init__(self, persist_directory="./chroma_data"):
        self.persist_directory = persist_directory
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load encoder: {e}")
            self.encoder = None

    def append_to_memory(self, user_id: str, new_info: str) -> str:
        """Embeds a manual note into the user's ChromaDB collection."""
        if not self.encoder:
            return "ระบบบันทึกความจำมีปัญหา (Encoder failed)"
            
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            collection = client.get_or_create_collection(name=f"drive_user_{user_id}")
            
            embedding = self.encoder.encode(new_info, convert_to_tensor=False).tolist()
            doc_id = f"manual_entry_{int(time.time())}"
            
            # Formatted exactly like the CSV parser so the dashboard can read it
            formatted_info = f"ข้อมูลเดือน: บันทึกใหม่\nชื่อพนักงาน: ระบบ/ผู้ใช้\nรายการเบิกวัสดุสิ้นเปลือง: {new_info}"
            
            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[{'user_id': user_id, 'source': 'manual_update', 'type': 'user_note'}],
                documents=[formatted_info]
            )
            return f"บันทึกข้อมูลลงในความจำเรียบร้อยแล้วครับ: '{new_info}'"
        except Exception as e:
            logger.error(f"DB Write Error: {e}")
            return f"เกิดข้อผิดพลาดในการบันทึก: {str(e)}"