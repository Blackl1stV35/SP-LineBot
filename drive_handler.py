#!/usr/bin/env python3
"""
Google Drive Handler: Recursive scan, dynamic folder creation, batch embedding.
Manages user document contexts with metadata tracking.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib

from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google.cloud import drive_v3
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import PyPDF2
from docx import Document

logger = logging.getLogger(__name__)

# ============================================================================
# GOOGLE DRIVE AUTHENTICATION & CLIENT
# ============================================================================

class DriveHandler:
    """Google Drive document handler with batch processing."""
    
    SUPPORTED_EXTENSIONS = {
        # Documents
        '.pdf': 'PDF',
        '.docx': 'Word',
        '.doc': 'Word',
        '.xlsx': 'Excel',
        '.xls': 'Excel',
        '.txt': 'Text',
        '.csv': 'CSV',
        # Images
        '.jpg': 'Image',
        '.jpeg': 'Image',
        '.png': 'Image',
        '.bmp': 'Image'
    }
    
    def __init__(self, service_account_json: str):
        self.service_account_json = service_account_json
        self.drive_service = None
        self.encoder = None
        self.chroma_client = None
        self.processed_files_log = Path('drive_sync/processed_files.json')
        self.user_folders = {}
        
        # Initialize
        self.authenticate()
        self._load_processed_files()
    
    def authenticate(self):
        """Authenticate with Google Drive."""
        try:
            if not os.path.exists(self.service_account_json):
                logger.error(f"Service account JSON not found: {self.service_account_json}")
                return False
            
            credentials = service_account.Credentials.from_service_account_file(
                self.service_account_json,
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            
            self.drive_service = build('drive', 'v3', credentials=credentials)
            
            # Initialize Chroma
            self.chroma_client = chromadb.Client()
            
            # Initialize encoder (sentence transformer)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info(f"✅ Google Drive authenticated (device={device})")
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def _load_processed_files(self):
        """Load metadata of already processed files."""
        try:
            if self.processed_files_log.exists():
                with open(self.processed_files_log, 'r') as f:
                    self.processed_files = json.load(f)
            else:
                self.processed_files = {}
                Path('drive_sync').mkdir(exist_ok=True)
        except Exception as e:
            logger.error(f"Load processed files failed: {e}")
            self.processed_files = {}
    
    def _save_processed_files(self):
        """Persist processed files metadata."""
        try:
            Path('drive_sync').mkdir(exist_ok=True)
            with open(self.processed_files_log, 'w') as f:
                json.dump(self.processed_files, f, indent=2)
        except Exception as e:
            logger.error(f"Save processed files failed: {e}")
    
    def create_user_folder(self, user_id: str, folder_name: str = None) -> bool:
        """Dynamically create folder for user in Drive."""
        try:
            if not self.drive_service:
                logger.error("Drive not authenticated")
                return False
            
            folder_name = folder_name or f"SP-LineBot-{user_id}"
            
            # Check if folder already exists
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.drive_service.files().list(q=query, spaces='drive', pageSize=1).execute()
            
            if results.get('files'):
                folder_id = results['files'][0]['id']
                logger.info(f"📁 User folder exists: {folder_id}")
            else:
                # Create new folder
                file_metadata = {
                    'name': folder_name,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                folder = self.drive_service.files().create(body=file_metadata, fields='id').execute()
                folder_id = folder.get('id')
                logger.info(f"✅ User folder created: {folder_id}")
            
            self.user_folders[user_id] = folder_id
            return True
        except Exception as e:
            logger.error(f"Create user folder failed: {e}")
            return False
    
    def scan_user_folder(self, user_id: str, folder_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Recursively scan user's folder and return document list.
        """
        try:
            if not self.drive_service:
                logger.error("Drive not authenticated")
                return []
            
            folder_id = folder_id or self.user_folders.get(user_id)
            if not folder_id:
                logger.warning(f"No folder for user {user_id}")
                return []
            
            logger.info(f"🔍 Scanning folder {folder_id} for user {user_id}")
            
            documents = []
            self._scan_folder_recursive(folder_id, documents, user_id)
            
            logger.info(f"✅ Found {len(documents)} documents for {user_id}")
            return documents
        except Exception as e:
            logger.error(f"Folder scan failed: {e}")
            return []
    
    def _scan_folder_recursive(self, folder_id: str, documents: List[Dict], user_id: str, depth: int = 0):
        """Recursively scan folder."""
        try:
            if depth > 5:  # Limit recursion depth
                logger.warning(f"Max recursion depth reached")
                return
            
            query = f"'{folder_id}' in parents and trashed=false"
            results = self.drive_service.files().list(
                q=query,
                spaces='drive',
                pageSize=100,
                fields='files(id, name, mimeType, size, modifiedTime)'
            ).execute()
            
            files = results.get('files', [])
            
            for file in files:
                file_id = file['id']
                name = file['name']
                mime_type = file['mimeType']
                size = int(file.get('size', 0))
                mod_time = file.get('modifiedTime')
                
                # Check if folder
                if mime_type == 'application/vnd.google-apps.folder':
                    logger.debug(f"  📁 Subfolder: {name}")
                    self._scan_folder_recursive(file_id, documents, user_id, depth + 1)
                else:
                    # Check if supported document type
                    ext = Path(name).suffix.lower()
                    if ext in self.SUPPORTED_EXTENSIONS:
                        file_hash = hashlib.md5(f"{file_id}_{mod_time}".encode()).hexdigest()
                        
                        # Skip if already processed
                        if file_hash in self.processed_files:
                            logger.debug(f"  ✓ Skipped (processed): {name}")
                            continue
                        
                        doc_type = self.SUPPORTED_EXTENSIONS[ext]
                        documents.append({
                            'id': file_id,
                            'name': name,
                            'type': doc_type,
                            'size_bytes': size,
                            'modified': mod_time,
                            'hash': file_hash,
                            'user_id': user_id
                        })
                        logger.info(f"  📄 Found: {name} ({doc_type}, {size} bytes)")
                    else:
                        logger.debug(f"  ⚠️  Skipped (unsupported): {name} ({ext})")
        except Exception as e:
            logger.error(f"Recursive scan failed: {e}")
    
    # ========================================================================
    # DOCUMENT EXTRACTION & EMBEDDING
    # ========================================================================
    
    def extract_text(self, file_id: str, file_name: str) -> str:
        """Extract text from document."""
        try:
            ext = Path(file_name).suffix.lower()
            
            # Download file
            request = self.drive_service.files().get_media(fileId=file_id)
            fh = open(f"temp_{file_id}", 'wb')
            downloader = MediaFileDownload(request)
            while not downloader.progress:
                pass
            fh.write(downloader.getbytes(downloader.size))
            fh.close()
            
            # Extract based on type
            if ext == '.pdf':
                text = self._extract_pdf(f"temp_{file_id}")
            elif ext in ['.docx', '.doc']:
                text = self._extract_docx(f"temp_{file_id}")
            elif ext == '.txt':
                with open(f"temp_{file_id}", 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            elif ext in ['.xlsx', '.xls']:
                text = self._extract_excel(f"temp_{file_id}")
            else:
                text = ""
            
            # Clean up
            os.remove(f"temp_{file_id}")
            
            logger.info(f"✅ Extracted {len(text)} chars from {file_name}")
            return text
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""
    
    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF."""
        try:
            text = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text())
            return '\n'.join(text)
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""
    
    def _extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX."""
        try:
            doc = Document(file_path)
            text = [paragraph.text for paragraph in doc.paragraphs]
            return '\n'.join(text)
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            return ""
    
    def _extract_excel(self, file_path: str) -> str:
        """Extract text from Excel."""
        try:
            import openpyxl
            wb = openpyxl.load_workbook(file_path, data_only=True)
            text = []
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                for row in ws.iter_rows(values_only=True):
                    text.append(' | '.join(str(cell or '') for cell in row))
            return '\n'.join(text)
        except Exception as e:
            logger.error(f"Excel extraction failed: {e}")
            return ""
    
    # ========================================================================
    # BATCH EMBEDDING
    # ========================================================================
    
    def batch_embed_documents(self, documents: List[Dict[str, Any]], user_id: str) -> int:
        """Batch embed and store documents in Chroma."""
        if not self.chroma_client or not self.encoder:
            logger.error("Chroma or encoder not initialized")
            return 0
        
        embedded_count = 0
        
        try:
            collection = self.chroma_client.get_or_create_collection(
                name=f"drive_user_{user_id}"
            )
            
            for doc in documents:
                try:
                    file_id = doc['id']
                    file_name = doc['name']
                    
                    # Extract text
                    text = self.extract_text(file_id, file_name)
                    if not text:
                        logger.warning(f"No text extracted from {file_name}")
                        continue
                    
                    # Chunk text (max 500 words)
                    chunks = self._chunk_text(text, chunk_size=500)
                    
                    # Embed chunks
                    for i, chunk in enumerate(chunks):
                        try:
                            embedding = self.encoder.encode(chunk, convert_to_tensor=False)
                            
                            doc_id = f"{user_id}_{file_id}_{i}"
                            collection.add(
                                ids=[doc_id],
                                embeddings=[embedding.tolist()],
                                metadatas=[{
                                    'user_id': user_id,
                                    'file_id': file_id,
                                    'file_name': file_name,
                                    'file_type': doc['type'],
                                    'chunk_idx': i,
                                    'timestamp': datetime.utcnow().isoformat()
                                }],
                                documents=[chunk]
                            )
                            embedded_count += 1
                        except Exception as e:
                            logger.error(f"Chunk embedding failed: {e}")
                    
                    # Mark as processed
                    self.processed_files[doc['hash']] = {
                        'file_id': file_id,
                        'name': file_name,
                        'processed_at': datetime.utcnow().isoformat()
                    }
                except Exception as e:
                    logger.error(f"Document embedding failed: {e}")
            
            # Save metadata
            self._save_processed_files()
            
            logger.info(f"✅ Embedded {embedded_count} chunks from {len(documents)} documents")
            return embedded_count
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return 0
    
    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks by word count."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks if chunks else [text]

# ============================================================================
# USER CONTEXT FETCH
# ============================================================================

def fetch_user_drive_context(user_id: str, drive_handler: Optional[DriveHandler] = None, 
                            top_k: int = 3) -> Dict[str, Any]:
    """
    Fetch user's Drive context (top-k relevant documents).
    """
    if not drive_handler or not drive_handler.chroma_client:
        logger.warning("Drive handler not initialized")
        return {}
    
    try:
        collection = drive_handler.chroma_client.get_or_create_collection(
            name=f"drive_user_{user_id}"
        )
        
        # Return collection summary
        count = collection.count()
        
        return {
            'user_id': user_id,
            'document_count': count,
            'status': 'ready' if count > 0 else 'empty'
        }
    except Exception as e:
        logger.error(f"Fetch context failed: {e}")
        return {'error': str(e)}
