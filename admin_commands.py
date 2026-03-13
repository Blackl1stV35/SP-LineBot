#!/usr/bin/env python3
"""
Admin Commands: Add/delete/list users, PIN-based auth, Drive folder auto-creation.
"""

import os
import json
import logging
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# ADMIN PIN AUTHENTICATION
# ============================================================================

ADMIN_PIN_HASH = os.getenv('ADMIN_PIN_HASH', hashlib.sha256('1234'.encode()).hexdigest())
USER_DATABASE = Path('drive_sync/users.json')

def verify_admin_pin(pin: str) -> bool:
    """Verify admin PIN (SHA256 hash)."""
    pin_hash = hashlib.sha256(pin.encode()).hexdigest()
    is_valid = pin_hash == ADMIN_PIN_HASH
    
    if not is_valid:
        logger.warning(f"Invalid admin PIN attempt")
    
    return is_valid

# ============================================================================
# USER DATABASE
# ============================================================================

def load_users() -> Dict[str, Dict[str, Any]]:
    """Load user database."""
    try:
        if USER_DATABASE.exists():
            with open(USER_DATABASE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Load users failed: {e}")
    
    return {}

def save_users(users: Dict[str, Dict[str, Any]]):
    """Save user database."""
    try:
        USER_DATABASE.parent.mkdir(parents=True, exist_ok=True)
        with open(USER_DATABASE, 'w') as f:
            json.dump(users, f, indent=2)
        logger.info(f"✅ Saved {len(users)} users")
    except Exception as e:
        logger.error(f"Save users failed: {e}")

# ============================================================================
# ADMIN COMMAND HANDLER
# ============================================================================

class AdminCommandHandler:
    """Execute admin commands with PIN auth."""
    
    def __init__(self, drive_handler):
        self.drive_handler = drive_handler
        self.users = load_users()
    
    def execute(self, intent: str, user_id: str, context: Dict[str, Any]) -> str:
        """Execute admin command."""
        
        # Extract PIN and args from context
        pin = context.get('pin', '')
        args = context.get('args', [])
        
        # Verify PIN
        if not verify_admin_pin(pin):
            logger.warning(f"Admin auth failed from {user_id}")
            return "❌ Invalid admin PIN"
        
        logger.info(f"✅ Admin auth: {intent} from {user_id}")
        
        # Dispatch command
        if intent == "ADMIN_ADD_USER":
            return self.add_user(args, user_id)
        elif intent == "ADMIN_DEL_USER":
            return self.delete_user(args, user_id)
        elif intent == "ADMIN_LIST_USERS":
            return self.list_users(user_id)
        else:
            return "Unknown admin command"
    
    # ========================================================================
    # ADD USER
    # ========================================================================
    
    def add_user(self, args: list, admin_user_id: str) -> str:
        """Add new user (sync with Drive folder)."""
        try:
            if not args or len(args) < 1:
                return "❌ Usage: add_user <user_id> [name]"
            
            new_user_id = args[0]
            user_name = args[1] if len(args) > 1 else new_user_id
            
            # Check if already exists
            if new_user_id in self.users:
                return f"⚠️  User {new_user_id} already exists"
            
            # Create Drive folder
            success = self.drive_handler.create_user_folder(new_user_id, f"SP-Bot-{user_name}")
            if not success:
                return f"❌ Failed to create Drive folder for {new_user_id}"
            
            # Add to database
            self.users[new_user_id] = {
                'name': user_name,
                'user_id': new_user_id,
                'added_by': admin_user_id,
                'created_at': datetime.utcnow().isoformat(),
                'status': 'active',
                'folder_id': self.drive_handler.user_folders.get(new_user_id)
            }
            
            save_users(self.users)
            
            logger.info(f"✅ User added: {new_user_id} ({user_name})")
            return f"✅ User {new_user_id} ({user_name}) added successfully!"
        except Exception as e:
            logger.error(f"Add user failed: {e}")
            return f"❌ Error: {str(e)}"
    
    # ========================================================================
    # DELETE USER
    # ========================================================================
    
    def delete_user(self, args: list, admin_user_id: str) -> str:
        """Delete user and optionally their Drive folder."""
        try:
            if not args or len(args) < 1:
                return "❌ Usage: delete_user <user_id>"
            
            user_id = args[0]
            
            # Check if exists
            if user_id not in self.users:
                return f"⚠️  User {user_id} not found"
            
            # Remove from database
            user_info = self.users.pop(user_id)
            save_users(self.users)
            
            logger.info(f"✅ User deleted: {user_id}")
            return f"✅ User {user_id} ({user_info['name']}) deleted successfully!"
        except Exception as e:
            logger.error(f"Delete user failed: {e}")
            return f"❌ Error: {str(e)}"
    
    # ========================================================================
    # LIST USERS
    # ========================================================================
    
    def list_users(self, admin_user_id: str) -> str:
        """List all users."""
        try:
            if not self.users:
                return "📭 No users found"
            
            user_list = "👥 **Current Users:**\n"
            for user_id, info in self.users.items():
                status = info['status']
                name = info.get('name', user_id)
                created = info['created_at'][:10]  # Date only
                user_list += f"• {name} ({user_id}) - {status} [added: {created}]\n"
            
            user_list += f"\n**Total:** {len(self.users)} users"
            
            logger.info(f"Listed {len(self.users)} users for {admin_user_id}")
            return user_list
        except Exception as e:
            logger.error(f"List users failed: {e}")
            return f"❌ Error: {str(e)}"
    
    # ========================================================================
    # BATCH ADD (for initialization)
    # ========================================================================
    
    def batch_add_users(self, user_list: list) -> str:
        """Batch add users from list."""
        try:
            results = []
            for user_info in user_list:
                user_id = user_info.get('user_id')
                name = user_info.get('name', user_id)
                
                if user_id in self.users:
                    results.append(f"⚠️  {user_id}: already exists")
                    continue
                
                # Create folder
                success = self.drive_handler.create_user_folder(user_id, f"SP-Bot-{name}")
                if not success:
                    results.append(f"❌ {user_id}: folder creation failed")
                    continue
                
                # Add to database
                self.users[user_id] = {
                    'name': name,
                    'user_id': user_id,
                    'added_by': 'batch_init',
                    'created_at': datetime.utcnow().isoformat(),
                    'status': 'active',
                    'folder_id': self.drive_handler.user_folders.get(user_id)
                }
                
                results.append(f"✅ {user_id}: added")
            
            save_users(self.users)
            
            summary = f"Batch add complete: {len([r for r in results if r.startswith('✅')])}/{len(user_list)} successful"
            logger.info(summary)
            return '\n'.join(results) + f"\n\n{summary}"
        except Exception as e:
            logger.error(f"Batch add failed: {e}")
            return f"❌ Batch add error: {str(e)}"
    
    # ========================================================================
    # USER CONTEXT
    # ========================================================================
    
    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get user context (profile, Drive status)."""
        try:
            if user_id not in self.users:
                return {"error": "User not found"}
            
            user_info = self.users[user_id]
            
            # Get Drive context
            drive_context = self.drive_handler.fetch_user_drive_context(user_id)
            
            return {
                **user_info,
                'drive': drive_context
            }
        except Exception as e:
            logger.error(f"Get context failed: {e}")
            return {"error": str(e)}

# ============================================================================
# INITIALIZE DEFAULT ADMIN PIN
# ============================================================================

def init_admin_pin(pin: str = '1234'):
    """Initialize admin PIN (call once on startup)."""
    pin_hash = hashlib.sha256(pin.encode()).hexdigest()
    print(f"🔐 Set ADMIN_PIN_HASH env var to: {pin_hash}")
    print(f"   Default PIN: {pin}")
    print(f"   Command: export ADMIN_PIN_HASH={pin_hash}")
