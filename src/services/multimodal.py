#!/usr/bin/env python3
"""
Multimodal Processing: Image OCR (Thai/Eng) and Voice STT.
Uses singleton embedder client for efficiency.
"""

import os
import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

import easyocr
import torch

from src.db.database import get_db_client, get_embedder_client

logger = logging.getLogger(__name__)


# ============================================================================
# IMAGE OCR (Thai + English)
# ============================================================================

def process_image_ocr(image_path: str, languages: List[str] = ['th', 'en']) -> Tuple[str, float]:
    """
    Extract text from image using EasyOCR.
    Returns extracted text and average confidence score.
    """
    try:
        if not os.path.exists(image_path):
            return "", 0.0
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        reader = easyocr.Reader(languages, gpu=(device == 'cuda'))
        
        result = reader.readtext(image_path, detail=1)
        
        texts = []
        confidences = []
        for (bbox, text, prob) in result:
            texts.append(text)
            confidences.append(prob)
        
        final_text = ' '.join(texts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        logger.info(f"OCR completed for {image_path} (confidence={avg_conf:.2f})")
        return final_text, avg_conf
    except Exception as e:
        logger.error(f"OCR failed for {image_path}: {e}")
        return "", 0.0


# ============================================================================
# IMAGE EMBEDDING (Using Singleton Client)
# ============================================================================

def extract_metadata_and_embed(image_path: str, text: str, user_id: str) -> bool:
    """
    Save extracted image text directly into ChromaDB.
    Uses singleton embedder client for efficiency.
    """
    try:
        db_client = get_db_client()
        embedder_client = get_embedder_client()
        
        if not db_client or not embedder_client:
            logger.error("Missing database or embedder client")
            return False
        
        collection = db_client.get_or_create_collection(name=f"drive_user_{user_id}")
        if not collection:
            return False
        
        # Use singleton embedder
        embedding = embedder_client.encode(text, convert_to_tensor=False)
        if embedding is None:
            return False
        
        doc_id = f"img_{Path(image_path).stem}"
        
        collection.add(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            metadatas=[{'user_id': user_id, 'source': 'image_upload', 'file_name': Path(image_path).name}],
            documents=[text]
        )
        logger.info(f"Image text embedded successfully for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Image embedding failed: {e}")
        return False


# ============================================================================
# VOICE PROCESSING (Speech-to-Text)
# ============================================================================

def process_voice_vosk(audio_path: str) -> str:
    """
    Speech-to-text using Google's SpeechRecognition API.
    Handles m4a format from LINE, converts to WAV, and transcribes.
    """
    try:
        import speech_recognition as sr
        from pydub import AudioSegment
        
        # Convert m4a to wav (LINE sends m4a format)
        wav_path = audio_path.replace('.m4a', '.wav')
        
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return ""
        
        try:
            audio = AudioSegment.from_file(audio_path)
            audio.export(wav_path, format="wav")
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return ""
        
        # Transcribe using SpeechRecognition
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
            
            # Thai language support
            text = recognizer.recognize_google(audio_data, language='th-TH')
            logger.info(f"Voice transcription successful: {text[:50]}...")
            return text
        except sr.RequestError:
            logger.warning("Google STT API unavailable, falling back to local methods")
            return ""
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return ""
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
    
    except Exception as e:
        logger.error(f"Voice transcription failed: {e}")
        return ""
