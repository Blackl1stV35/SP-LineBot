#!/usr/bin/env python3
"""
Admin Commands: Add/delete/list users, PIN-based auth, Drive folder auto-creation.
"""

import os
import json
import logging
import hashlib
import re
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# ADMIN PIN AUTHENTICATION
# ============================================================================

USER_DATABASE = Path('drive_sync/users.json')

def verify_admin_pin(pin: str) -> bool:
    """Verify admin PIN (SHA256 hash). Evaluated dynamically at runtime."""
    expected_hash = os.getenv('ADMIN_PIN_HASH', hashlib.sha256('8899'.encode()).hexdigest()).strip().strip('"').strip("'")
    pin_hash = hashlib.sha256(pin.encode()).hexdigest()
    
    is_valid = pin_hash == expected_hash
    
    if not is_valid:
        logger.warning(f"Invalid admin PIN attempt. Expected: {expected_hash[:10]}... Got: {pin_hash[:10]}...")
    
    return is_valid

# ============================================================================
# USER DATABASE
# ============================================================================

def load_users() -> Dict[str, Dict[str, Any]]:
    try:
        if USER_DATABASE.exists():
            with open(USER_DATABASE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Load users failed: {e}")
    return {}

def save_users(users: Dict[str, Dict[str, Any]]):
    try:
        USER_DATABASE.parent.mkdir(parents=True, exist_ok=True)
        with open(USER_DATABASE, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=2)
        logger.info(f"Saved {len(users)} users")
    except Exception as e:
        logger.error(f"Save users failed: {e}")

# ============================================================================
# ADMIN COMMAND HANDLER
# ============================================================================

class AdminCommandHandler:
    def __init__(self, drive_handler):
        self.drive_handler = drive_handler
        self.users = load_users()
    
    def execute(self, intent: str, user_id: str, text: str, context: Dict[str, Any]) -> str:
        pin = context.get('pin', '')
        if not pin:
            words = text.split()
            for word in words:
                if word.isdigit() and len(word) >= 4:
                    pin = word
                    break
        
        if not verify_admin_pin(pin):
            logger.warning(f"Admin auth failed from {user_id}")
            return "Invalid admin PIN. Please start your command with your 4-digit PIN (e.g., '1234 add user...')."
        
        logger.info(f"Admin auth: {intent} from {user_id}")
        
        if intent == "ADMIN_ADD_USER":
            return self.add_user(text, user_id)
        elif intent == "ADMIN_DEL_USER":
            return self.delete_user(text, user_id)
        elif intent == "ADMIN_LIST_USERS":
            return self.list_users(user_id)
        else:
            return "Unknown admin command"
    
    def add_user(self, text: str, admin_user_id: str) -> str:
        try:
            parts = text.split()
            target_line_id = None
            target_email = None

            for part in parts:
                if "@" in part and "." in part:
                    target_email = part
                elif part.lower().startswith("u") and len(part) > 10:
                    target_line_id = part.upper()
            
            if not target_line_id or not target_email:
                return "Invalid format. Please use: '[PIN] add user [Line_ID] [Email_Address]'"
            
            if target_line_id in self.users:
                return f"User {target_line_id} already exists"
            
            success, folder_link = self.drive_handler.create_user_folder(
                user_id=target_line_id, 
                user_email=target_email
            )
            
            if not success:
                return f"Failed to create or share Drive folder for {target_line_id}. Check terminal logs."
            
            self.users[target_line_id] = {
                'email': target_email,
                'user_id': target_line_id,
                'added_by': admin_user_id,
                'created_at': datetime.utcnow().isoformat(),
                'status': 'active',
                'folder_id': self.drive_handler.user_folders.get(target_line_id)
            }
            
            save_users(self.users)
            
            logger.info(f"User added: {target_line_id} ({target_email})")
            return (f"User {target_line_id} registered!\n\n"
                    f"Folder created and shared with {target_email}. "
                    f"They have been emailed an invitation link by Google Drive.\n\n"
                    f"Direct Link: {folder_link}")
        except Exception as e:
            logger.error(f"Add user failed: {e}")
            return f"Error: {str(e)}"
    
    def delete_user(self, text: str, admin_user_id: str) -> str:
        try:
            parts = text.split()
            target_line_id = None
            for part in parts:
                if part.lower().startswith("u") and len(part) > 10:
                    target_line_id = part.upper()
                    break

            if not target_line_id:
                return "Usage: [PIN] delete user <user_id>"
            
            if target_line_id not in self.users:
                return f"User {target_line_id} not found"
            
            user_info = self.users.pop(target_line_id)
            save_users(self.users)
            
            logger.info(f"User deleted: {target_line_id}")
            return f"User {target_line_id} ({user_info.get('email', 'Unknown')}) deleted successfully!"
        except Exception as e:
            logger.error(f"Delete user failed: {e}")
            return f"Error: {str(e)}"
    
    def list_users(self, admin_user_id: str) -> str:
        try:
            if not self.users:
                return "No users found"
            
            user_list = "**Current Users:**\n"
            for user_id, info in self.users.items():
                status = info['status']
                email = info.get('email', user_id)
                created = info['created_at'][:10]
                user_list += f"• {email} ({user_id}) - {status} [added: {created}]\n"
            
            user_list += f"\n**Total:** {len(self.users)} users"
            
            logger.info(f"Listed {len(self.users)} users for {admin_user_id}")
            return user_list
        except Exception as e:
            logger.error(f"List users failed: {e}")
            return f"Error: {str(e)}"
    
    def batch_add_users(self, user_list: list) -> str:
        try:
            results = []
            for user_info in user_list:
                user_id = user_info.get('user_id')
                email = user_info.get('email', f"{user_id}@example.com")
                
                if user_id in self.users:
                    results.append(f"{user_id}: already exists")
                    continue
                
                success, folder_link = self.drive_handler.create_user_folder(user_id, email)
                if not success:
                    results.append(f"{user_id}: folder creation failed")
                    continue
                
                self.users[user_id] = {
                    'email': email,
                    'user_id': user_id,
                    'added_by': 'batch_init',
                    'created_at': datetime.utcnow().isoformat(),
                    'status': 'active',
                    'folder_id': self.drive_handler.user_folders.get(user_id)
                }
                results.append(f"{user_id}: added")
            
            save_users(self.users)
            summary = f"Batch add complete: {len([r for r in results if r.startswith('')])}/{len(user_list)} successful"
            logger.info(summary)
            return '\n'.join(results) + f"\n\n{summary}"
        except Exception as e:
            logger.error(f"Batch add failed: {e}")
            return f"Batch add error: {str(e)}"
    
    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        try:
            user_info = None
            for stored_id, data in self.users.items():
                if stored_id.lower() == user_id.lower():
                    user_info = data
                    break
                    
            if not user_info:
                return {"error": "User not found"}
            
            drive_context = self.drive_handler.fetch_user_drive_context(user_id)
            
            return {
                **user_info,
                'drive': drive_context
            }
        except Exception as e:
            logger.error(f"Get context failed: {e}")
            return {"error": str(e)}

def init_admin_pin(pin: str = '1234'):
    pin_hash = hashlib.sha256(pin.encode()).hexdigest()
    print(f"Set ADMIN_PIN_HASH env var to: {pin_hash}")
    print(f"   Default PIN: {pin}")
    print(f"   Command: export ADMIN_PIN_HASH={pin_hash}")