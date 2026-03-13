#!/usr/bin/env python3
"""
Multimodal Processing: Full phases for image (OCR Thai/Eng) and voice (Vosk STT).
Metadata extraction, embedding, and reinforcement learning suggestions.
"""

import os
import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any
import threading
from datetime import datetime

import easyocr
import vosk
import soundfile as sf
import numpy as np
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ============================================================================
# IMAGE OCR (Thai + English)
# ============================================================================

def process_image_ocr(image_path: str, languages: List[str] = ['th', 'en']) -> Tuple[str, float]:
    """
    Extract text from image using EasyOCR (Thai + English).
    Returns extracted text and confidence score.
    """
    try:
        logger.info(f"🖼️  Processing image: {image_path}")
        
        # Validate image exists
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return "", 0.0
        
        # Initialize reader (GPU/CPU auto)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        reader = easyocr.Reader(languages, gpu=(device == 'cuda'))
        
        # Read image
        results = reader.readtext(image_path)
        
        if not results:
            logger.warning("No text detected in image")
            return "", 0.0
        
        # Extract text & confidence
        texts = [result[1] for result in results]
        confidences = [result[2] for result in results]
        
        full_text = '\n'.join(texts)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        logger.info(f"✅ OCR extracted {len(texts)} lines (conf={avg_confidence:.2f})")
        logger.debug(f"Text sample: {full_text[:100]}...")
        
        return full_text, float(avg_confidence)
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        return "", 0.0

# ============================================================================
# VOICE STT (Vosk)
# ============================================================================

def process_voice_vosk(audio_path: str) -> str:
    """
    Convert speech to text using Vosk (offline, free).
    Supports multiple languages (Thai model via lang file).
    """
    try:
        logger.info(f"🎤 Processing voice: {audio_path}")
        
        # Validate audio file
        if not os.path.exists(audio_path):
            logger.error(f"Audio not found: {audio_path}")
            return ""
        
        # Initialize Vosk recognizer
        # Using English model (no Thai offline STT)
        vosk_model_path = os.getenv('VOSK_MODEL_PATH', './vosk_models/en-us')
        
        if not os.path.exists(vosk_model_path):
            logger.warning(f"Vosk model not found: {vosk_model_path}")
            logger.info("Download from: https://alphacephei.com/vosk/models")
            return ""
        
        model = vosk.Model(vosk_model_path)
        recognizer = vosk.KaldiRecognizer(model, 16000)
        recognizer.SetWords(['inventory', 'stock', 'repair', 'part', 'service'])
        
        # Load audio
        data, samplerate = sf.read(audio_path)
        
        # Resample if needed
        if samplerate != 16000:
            from scipy import signal
            num_samples = round(len(data) * 16000 / samplerate)
            data = signal.resample(data, num_samples)
            samplerate = 16000
        
        # Convert to 16-bit PCM
        audio_data = (data * 32768).astype(np.int16).tobytes()
        
        # Process audio
        results = []
        for i in range(0, len(audio_data), 4000):
            recognizer.AcceptWaveform(audio_data[i:i+4000])
            
            partial = recognizer.PartialResult()
            if partial:
                logger.debug(f"Partial: {partial}")
        
        # Final result
        final = recognizer.FinalResult()
        result_json = json.loads(final)
        text = result_json.get('result', [])
        
        if text:
            full_text = ' '.join([word['conf'] for word in text])
            logger.info(f"✅ STT extracted: {full_text}")
            return full_text
        else:
            partial = json.loads(recognizer.PartialResult())
            text = partial.get('partial', '')
            logger.info(f"✅ STT partial: {text}")
            return text
    except Exception as e:
        logger.error(f"Voice processing failed: {e}")
        return ""

# ============================================================================
# METADATA EXTRACTION & EMBEDDING
# ============================================================================

class MetadataExtractor:
    """Extract metadata from images (OCR, EXIF, dimensions)."""
    
    @staticmethod
    def extract_image_metadata(image_path: str) -> Dict[str, Any]:
        """Extract image metadata."""
        try:
            img = Image.open(image_path)
            
            metadata = {
                "filename": Path(image_path).name,
                "size_bytes": os.path.getsize(image_path),
                "dimensions": img.size,
                "format": img.format,
                "mode": img.mode,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # EXIF data
            try:
                from PIL.Image import Exif
                exif = img.getexif()
                if exif:
                    metadata["exif"] = {
                        "make": exif.get(271, "Unknown"),
                        "datetime": exif.get(306, "Unknown")
                    }
            except:
                pass
            
            return metadata
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return {}

def extract_metadata_and_embed(file_path: str, extracted_text: str, user_id: str) -> Dict[str, Any]:
    """
    Extract metadata, embed text + metadata, store in Chroma.
    """
    try:
        logger.info(f"🔗 Embedding content from {file_path}")
        
        # Extract metadata
        metadata = MetadataExtractor.extract_image_metadata(file_path)
        metadata['user_id'] = user_id
        metadata['extracted_text'] = extracted_text[:500]
        
        # Initialize Chroma
        import chromadb
        client = chromadb.Client()
        collection = client.get_or_create_collection(name=f"user_{user_id}")
        
        # Get sentence transformer for embedding
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Embed text
        embedding = encoder.encode(extracted_text, convert_to_tensor=False)
        
        # Store in Chroma
        doc_id = f"{user_id}_{Path(file_path).stem}"
        collection.add(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            documents=[extracted_text]
        )
        
        logger.info(f"✅ Embedded {doc_id} in Chroma")
        return metadata
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return {}

# ============================================================================
# REINFORCEMENT LEARNING SUGGESTIONS
# ============================================================================

class RLSuggestionEngine:
    """Suggest actions based on extracted content (RL-inspired)."""
    
    def __init__(self):
        self.action_rewards = {
            "inventory_lookup": 1.0,
            "order_part": 0.8,
            "schedule_service": 0.9,
            "update_status": 0.7,
            "escalate_to_tech": 0.6
        }
    
    def suggest_action(self, extracted_text: str, user_context: Dict[str, Any]) -> str:
        """Suggest next action based on content."""
        try:
            text_lower = extracted_text.lower()
            
            # Pattern matching for suggestions
            if any(kw in text_lower for kw in ['stock', 'qty', 'quantity', 'availability']):
                return "inventory_lookup"
            elif any(kw in text_lower for kw in ['order', 'buy', 'purchase', 'need']):
                return "order_part"
            elif any(kw in text_lower for kw in ['repair', 'fix', 'broken', 'issue']):
                return "schedule_service"
            elif any(kw in text_lower for kw in ['update', 'change', 'modify', 'status']):
                return "update_status"
            else:
                return "escalate_to_tech"
        except Exception as e:
            logger.error(f"Suggestion failed: {e}")
            return "escalate_to_tech"

# ============================================================================
# THREADED PROCESSING FOR BURST HANDLING
# ============================================================================

def process_burst_images(image_paths: List[str], user_id: str, callback=None):
    """Thread-based processing for multiple images (scalability)."""
    threads = []
    results = []
    
    def worker(img_path, idx):
        try:
            text, conf = process_image_ocr(img_path)
            metadata = extract_metadata_and_embed(img_path, text, user_id)
            results.append({"index": idx, "status": "ok", "metadata": metadata})
            
            if callback:
                callback(idx, metadata)
        except Exception as e:
            logger.error(f"Worker failed for {img_path}: {e}")
            results.append({"index": idx, "status": "error", "error": str(e)})
    
    # Process up to 5 images in parallel (resource limit)
    for i, img_path in enumerate(image_paths[:5]):
        t = threading.Thread(target=worker, args=(img_path, i))
        t.start()
        threads.append(t)
    
    # Wait for all threads
    for t in threads:
        t.join(timeout=60)
    
    logger.info(f"✅ Burst processing complete: {len(results)}/{len(image_paths)} images")
    return results

# ============================================================================
# CLEANUP & TEMP FILE MANAGEMENT
# ============================================================================

def cleanup_temp_files(directory: str, max_age_hours: int = 24):
    """Clean up temporary files older than max_age_hours."""
    try:
        import shutil
        from datetime import datetime, timedelta
        
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        for file in Path(directory).glob('*'):
            if file.is_file():
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                if mtime < cutoff:
                    file.unlink()
                    logger.info(f"🗑️  Deleted temp file: {file.name}")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
