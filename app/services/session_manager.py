"""
Session Manager - Core State Persistence

Provides fault-tolerant session tracking with auto-save and recovery.
Every processed row is saved immediately to prevent data loss.

Design Principles:
- Atomic writes (temp file → rename)
- Auto-save after each row
- Crash recovery support
- Thread-safe operations
"""

import json
import os
import logging
import threading
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """Represents the current session state."""
    session_id: str
    created_at: str
    updated_at: str
    processed_count: int = 0
    rows: List[Dict[str, Any]] = field(default_factory=list)
    current_field_index: int = 0
    is_complete: bool = False


class SessionManager:
    """
    Manages session persistence with auto-save and crash recovery.
    
    Responsibilities:
    - Track current session state
    - Auto-save after each row (not at batch end)
    - Atomic writes to prevent corruption
    - Recovery of incomplete sessions
    
    Thread-safe for concurrent access.
    """
    
    def __init__(self, storage_path: str = "data/sessions"):
        """
        Initialize the session manager.
        
        Args:
            storage_path: Path to session storage directory
        """
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Current session state
        self._current_session: Optional[SessionState] = None
        self._session_file: Optional[Path] = None
        
        logger.info(f"SessionManager initialized: {self._storage_path}")
    
    # =========================================================================
    # Session Lifecycle
    # =========================================================================
    
    def create_session(self) -> str:
        """
        Create a new session.
        
        Returns:
            New session ID
        """
        with self._lock:
            # Generate session ID
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create session state
            now = datetime.now().isoformat()
            self._current_session = SessionState(
                session_id=session_id,
                created_at=now,
                updated_at=now
            )
            
            # Create session file path
            self._session_file = self._storage_path / f"{session_id}.json"
            
            # Save initial state
            self._save_session()
            
            logger.info(f"Created new session: {session_id}")
            return session_id
    
    def get_current_session_id(self) -> Optional[str]:
        """Get the current session ID."""
        with self._lock:
            return self._current_session.session_id if self._current_session else None
    
    # =========================================================================
    # Row Operations (Auto-Save)
    # =========================================================================
    
    def save_row(self, row: Dict[str, Any]) -> bool:
        """
        Save a single row and auto-save to disk.
        
        Args:
            row: Row data to save
            
        Returns:
            True if saved successfully
        """
        with self._lock:
            if not self._current_session:
                logger.warning("No active session, creating one")
                self.create_session()
            
            try:
                # Add row to session
                self._current_session.rows.append(row)
                self._current_session.processed_count += 1
                self._current_session.updated_at = datetime.now().isoformat()
                
                # Atomic write to disk
                self._save_session()
                
                logger.debug(f"Saved row {self._current_session.processed_count}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to save row: {e}", exc_info=True)
                return False
    
    def update_current_field(self, field_index: int) -> None:
        """
        Update the current field index (for UI state recovery).
        
        Args:
            field_index: Current field position
        """
        with self._lock:
            if self._current_session:
                self._current_session.current_field_index = field_index
                self._save_session()
    
    def mark_complete(self) -> None:
        """Mark the current session as complete."""
        with self._lock:
            if self._current_session:
                self._current_session.is_complete = True
                self._current_session.updated_at = datetime.now().isoformat()
                self._save_session()
                logger.info(f"Session marked complete: {self._current_session.session_id}")
    
    # =========================================================================
    # Recovery System
    # =========================================================================
    
    def load_last_session(self) -> Optional[Dict[str, Any]]:
        """
        Load the last incomplete session for recovery.
        
        Returns:
            Session data dict or None if no incomplete session
        """
        with self._lock:
            try:
                # Find all session files
                session_files = sorted(self._storage_path.glob("session_*.json"))
                
                if not session_files:
                    logger.info("No session files found")
                    return None
                
                # Load most recent incomplete session
                for session_file in reversed(session_files):
                    try:
                        with open(session_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Check if incomplete
                        if not data.get('is_complete', False):
                            # Restore session state
                            self._current_session = SessionState(
                                session_id=data['session_id'],
                                created_at=data['created_at'],
                                updated_at=data['updated_at'],
                                processed_count=data.get('processed_count', 0),
                                rows=data.get('rows', []),
                                current_field_index=data.get('current_field_index', 0),
                                is_complete=False
                            )
                            self._session_file = session_file
                            
                            logger.info(
                                f"Recovered session: {self._current_session.session_id} "
                                f"with {self._current_session.processed_count} rows"
                            )
                            return {
                                'session_id': self._current_session.session_id,
                                'rows': self._current_session.rows,
                                'current_field_index': self._current_session.current_field_index,
                                'processed_count': self._current_session.processed_count
                            }
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Corrupted session file: {session_file.name} - {e}")
                        continue
                
                logger.info("No incomplete sessions found")
                return None
                
            except Exception as e:
                logger.error(f"Failed to load last session: {e}", exc_info=True)
                return None
    
    def get_recovery_candidates(self) -> List[Dict[str, Any]]:
        """
        Get all recoverable sessions.
        
        Returns:
            List of session info dicts
        """
        with self._lock:
            candidates = []
            
            try:
                for session_file in sorted(self._storage_path.glob("session_*.json")):
                    try:
                        with open(session_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        if not data.get('is_complete', False):
                            candidates.append({
                                'session_id': data['session_id'],
                                'created_at': data['created_at'],
                                'processed_count': data.get('processed_count', 0),
                                'file_path': str(session_file)
                            })
                    except Exception:
                        continue
                        
            except Exception as e:
                logger.error(f"Failed to get recovery candidates: {e}")
            
            return candidates
    
    # =========================================================================
    # Atomic Write Operations
    # =========================================================================
    
    def _save_session(self) -> None:
        """
        Atomic write: write to temp file then rename.
        
        Raises:
            IOError: If write fails
        """
        if not self._current_session or not self._session_file:
            return
        
        # Create temp file in same directory
        temp_file = self._session_file.with_suffix('.tmp')
        
        try:
            # Write to temp file
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self._current_session), f, indent=2)
            
            # Atomic rename
            shutil.move(str(temp_file), str(self._session_file))
            
        except Exception as e:
            # Clean up temp file if exists
            if temp_file.exists():
                temp_file.unlink()
            raise IOError(f"Failed to save session: {e}")
    
    # =========================================================================
    # Session Queries
    # =========================================================================
    
    def get_row_count(self) -> int:
        """Get number of saved rows in current session."""
        with self._lock:
            return len(self._current_session.rows) if self._current_session else 0
    
    def get_all_rows(self) -> List[Dict[str, Any]]:
        """Get all rows from current session."""
        with self._lock:
            return self._current_session.rows.copy() if self._current_session else []
    
    def get_session_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of current session."""
        with self._lock:
            if not self._current_session:
                return None
            
            return {
                'session_id': self._current_session.session_id,
                'created_at': self._current_session.created_at,
                'processed_count': self._current_session.processed_count,
                'is_complete': self._current_session.is_complete,
                'row_count': len(self._current_session.rows)
            }
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    
    def clear_session(self) -> None:
        """Clear current session from memory (keep file)."""
        with self._lock:
            self._current_session = None
            self._session_file = None
            logger.info("Session cleared from memory")
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a specific session file.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if deleted
        """
        with self._lock:
            try:
                session_file = self._storage_path / f"{session_id}.json"
                if session_file.exists():
                    session_file.unlink()
                    logger.info(f"Deleted session: {session_id}")
                    return True
                return False
            except Exception as e:
                logger.error(f"Failed to delete session: {e}")
                return False
    
    def clear_all_sessions(self) -> int:
        """
        Clear all session files.
        
        Returns:
            Number of files deleted
        """
        with self._lock:
            count = 0
            for session_file in self._storage_path.glob("session_*.json"):
                try:
                    session_file.unlink()
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {session_file.name}: {e}")
            
            self._current_session = None
            self._session_file = None
            logger.info(f"Cleared {count} session files")
            return count