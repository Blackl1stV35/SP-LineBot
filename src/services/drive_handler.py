import os
import logging
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/drive']

class DriveHandler:
    def __init__(self):
        self.service_account_file = os.getenv("SERVICE_ACCOUNT_FILE", "service_account.json")
        try:
            creds = Credentials.from_service_account_file(self.service_account_file, scopes=SCOPES)
            self.service = build('drive', 'v3', credentials=creds)
            logger.info("Google Drive API initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Drive API: {e}")
            self.service = None

    def create_user_folder(self, user_id: str) -> tuple[str, str]:
        """Creates a public writer folder for the user and returns (Link, FolderID)"""
        if not self.service:
            raise Exception("Drive service not initialized.")

        # 1. Create Folder
        folder_name = f"SP_Auto_Service_User_{user_id[-5:]}"
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        folder = self.service.files().create(body=file_metadata, fields='id, webViewLink').execute()
        folder_id = folder.get('id')
        link = folder.get('webViewLink')

        # 2. Grant "Anyone with the link can edit" permission
        permission = {'type': 'anyone', 'role': 'writer'}
        self.service.permissions().create(fileId=folder_id, body=permission).execute()

        logger.info(f"Created frictionless folder for {user_id}: {folder_id}")
        return link, folder_id