"""
Self Notes - Persistent behavioral notes that the bot can read and write.

Allows the bot to "learn" from user feedback by storing behavioral adjustments,
preferences, and habits in a JSON file that gets injected into the system prompt.
"""

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional
from loguru import logger


NOTES_FILE = "self_notes.json"
MAX_NOTES = 50


class SelfNotes:
    """Persistent key-value store for the bot's self-discovered behavioral notes."""

    def __init__(self, data_dir: str = "."):
        self._file_path = os.path.join(data_dir, NOTES_FILE)
        self._notes: List[Dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self._file_path):
            try:
                with open(self._file_path, "r", encoding="utf-8") as f:
                    self._notes = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load self notes: {e}")
                self._notes = []
        logger.info(f"SelfNotes: loaded {len(self._notes)} notes")

    def _save(self):
        try:
            with open(self._file_path, "w", encoding="utf-8") as f:
                json.dump(self._notes, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save self notes: {e}")

    def add_note(self, content: str, category: str = "habit") -> str:
        """Add a behavioral note. Returns confirmation message."""
        if not content.strip():
            return ""

        for existing in self._notes:
            if existing["content"] == content.strip():
                return "already_exists"

        note = {
            "content": content.strip(),
            "category": category,
            "created_at": time.time(),
            "created_readable": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        self._notes.append(note)

        if len(self._notes) > MAX_NOTES:
            self._notes = self._notes[-MAX_NOTES:]

        self._save()
        logger.info(f"SelfNote added [{category}]: {content[:60]}")
        return "added"

    def remove_note(self, keyword: str) -> bool:
        """Remove notes containing the keyword."""
        before = len(self._notes)
        self._notes = [n for n in self._notes if keyword not in n["content"]]
        if len(self._notes) < before:
            self._save()
            logger.info(f"SelfNote removed {before - len(self._notes)} notes matching '{keyword}'")
            return True
        return False

    def get_all_notes(self) -> List[Dict]:
        return self._notes.copy()

    def build_prompt_section(self) -> str:
        """Build a prompt section from all notes for injection into system prompt."""
        if not self._notes:
            return ""

        lines = ["[你的行为备忘录 - 这些是你自己记录的习惯和用户偏好]"]
        for note in self._notes:
            lines.append(f"- {note['content']}")
        lines.append("[备忘录结束]")
        return "\n".join(lines)
