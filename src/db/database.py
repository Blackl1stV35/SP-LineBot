#!/usr/bin/env python3
"""
Singleton Database Client for ChromaDB and SentenceTransformers.
Optimizes memory and I/O by ensuring a single instance across the entire application.
"""

import logging
import threading
from typing import Optional
import chromadb
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

_db_lock = threading.Lock()
_embedder_lock = threading.Lock()

class DatabaseClient:
    """Singleton wrapper for ChromaDB PersistentClient."""
    _instance: Optional['DatabaseClient'] = None
    
    def __new__(cls, persist_directory: str = "./chroma_data"):
        if cls._instance is None:
            with _db_lock:
                if cls._instance is None:
                    cls._instance = super(DatabaseClient, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, persist_directory: str = "./chroma_data"):
        if self._initialized:
            return
        
        self.persist_directory = persist_directory
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            logger.info(f"ChromaDB initialized (path={persist_directory})")
            self._initialized = True
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            self.client = None
            self._initialized = True
    
    def get_or_create_collection(self, name: str):
        """Get or create a named collection."""
        if not self.client:
            return None
        try:
            return self.client.get_or_create_collection(name=name)
        except Exception as e:
            logger.error(f"Failed to get/create collection '{name}': {e}")
            return None
    
    def get_collection(self, name: str):
        """Get an existing collection."""
        if not self.client:
            return None
        try:
            return self.client.get_collection(name=name)
        except Exception as e:
            logger.debug(f"Collection '{name}' not found: {e}")
            return None
    
    def list_collections(self):
        """List all collections."""
        if not self.client:
            return []
        try:
            return self.client.list_collections()
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []


class EmbedderClient:
    """Singleton wrapper for SentenceTransformer embeddings."""
    _instance: Optional['EmbedderClient'] = None
    
    def __new__(cls, model_name: str = 'all-MiniLM-L6-v2'):
        if cls._instance is None:
            with _embedder_lock:
                if cls._instance is None:
                    cls._instance = super(EmbedderClient, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        if self._initialized:
            return
        
        self.model_name = model_name
        self.encoder = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            self.encoder = SentenceTransformer(model_name)
            logger.info(f"SentenceTransformer loaded (model={model_name}, device={self.device})")
            self._initialized = True
        except Exception as e:
            logger.error(f"SentenceTransformer initialization failed: {e}")
            self._initialized = True
    
    def encode(self, text: str, convert_to_tensor: bool = False):
        """Encode text to embeddings."""
        if not self.encoder:
            return None
        try:
            return self.encoder.encode(text, convert_to_tensor=convert_to_tensor)
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            return None


# Global singleton instances
_db_client: Optional[DatabaseClient] = None
_embedder_client: Optional[EmbedderClient] = None


def get_db_client(persist_directory: str = "./chroma_data") -> DatabaseClient:
    """Get the singleton database client."""
    global _db_client
    if _db_client is None:
        _db_client = DatabaseClient(persist_directory)
    return _db_client


def get_embedder_client(model_name: str = 'all-MiniLM-L6-v2') -> EmbedderClient:
    """Get the singleton embedder client."""
    global _embedder_client
    if _embedder_client is None:
        _embedder_client = EmbedderClient(model_name)
    return _embedder_client
