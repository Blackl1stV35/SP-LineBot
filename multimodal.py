#!/usr/bin/env python3
"""
Multimodal Processing: Full phases for image (OCR Thai/Eng) and voice (Vosk STT).
Metadata extraction and embedding.
"""

import os
import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

import easyocr
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ============================================================================
# IMAGE OCR (Thai + English)
# ============================================================================

def process_image_ocr(image_path: str, languages: List[str] = ['th', 'en']) -> Tuple[str, float]:
    """Extract text from image using EasyOCR."""
    try:
        if not os.path.exists(image_path):
            return "", 0.0
            
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        reader = easyocr.Reader(languages, gpu=(device == 'cuda'))
        
        # Detail=1 gives bounding boxes, text, and confidence
        result = reader.readtext(image_path, detail=1)
        
        texts = []
        confidences = []
        for (bbox, text, prob) in result:
            texts.append(text)
            confidences.append(prob)
            
        final_text = ' '.join(texts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        return final_text, avg_conf
    except Exception as e:
        logger.error(f"OCR failed for {image_path}: {e}")
        return "", 0.0

# ============================================================================
# METADATA & CHROMA DB EMBEDDING FOR IMAGES
# ============================================================================

def extract_metadata_and_embed(image_path: str, text: str, user_id: str):
    """Save the text extracted from the image directly into ChromaDB."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_data")
        collection = client.get_or_create_collection(name=f"drive_user_{user_id}")
        
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = encoder.encode(text, convert_to_tensor=False).tolist()
        
        doc_id = f"img_{Path(image_path).stem}"
        
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[{'user_id': user_id, 'source': 'image_upload', 'file_name': Path(image_path).name}],
            documents=[text]
        )
        logger.info(f"Image text embedded successfully into user collection.")
    except Exception as e:
        logger.error(f"Image embedding failed: {e}")

# ============================================================================
# VOICE PROCESSING (VOSK)
# ============================================================================

def process_voice_vosk(audio_path: str) -> str:
    """
    Speech-to-text using Vosk or SpeechRecognition.
    For production, we use a cloud fallback or light Vosk model.
    """
    try:
        import speech_recognition as sr
        
        # Convert m4a to wav via pydub if necessary (Line sends m4a usually)
        from pydub import AudioSegment
        wav_path = audio_path.replace('.m4a', '.wav')
        
        audio = AudioSegment.from_file(audio_path)
        audio.export(wav_path, format="wav")
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            
        # Try Google's free API for STT as a robust fallback to Vosk
        text = recognizer.recognize_google(audio_data, language='th-TH')
        
        if os.path.exists(wav_path):
            os.remove(wav_path)
            
        return text
    except Exception as e:
        logger.error(f"Voice transcription failed: {e}")
        return ""