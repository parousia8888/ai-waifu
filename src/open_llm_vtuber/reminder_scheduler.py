"""
Reminder Scheduler - Enables proactive reminders and time-based actions.

Stores reminders persistently in a JSON file, runs a background async loop
that checks for due reminders every 30 seconds, and triggers the bot to
speak through the existing conversation pipeline.
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, Any, List
from loguru import logger


REMINDERS_FILE = "reminders.json"

RELATIVE_TIME_PATTERNS = [
    (re.compile(r"(\d+)\s*分钟[后以]"), "minutes"),
    (re.compile(r"(\d+)\s*小时[后以]"), "hours"),
    (re.compile(r"(\d+)\s*秒[后以]"), "seconds"),
    (re.compile(r"半\s*小时[后以]"), "half_hour"),
    (re.compile(r"(\d+)\s*minutes?\s*(?:later|from now)", re.IGNORECASE), "minutes"),
    (re.compile(r"(\d+)\s*hours?\s*(?:later|from now)", re.IGNORECASE), "hours"),
]

ABSOLUTE_TIME_PATTERNS = [
    re.compile(r"(?:明天|明日)\s*(?:上午|下午|晚上)?\s*(\d{1,2})[:\uff1a](\d{2})"),
    re.compile(r"(?:明天|明日)\s*(?:上午|下午|晚上)?\s*(\d{1,2})\s*[点時时](?:\s*(\d{1,2})\s*分)?"),
    re.compile(r"(?:今天|今日)?\s*(?:上午|下午|晚上)?\s*(\d{1,2})[:\uff1a](\d{2})"),
    re.compile(r"(?:今天|今日)?\s*(?:上午|下午|晚上)?\s*(\d{1,2})\s*[点時时](?:\s*(\d{1,2})\s*分)?"),
    re.compile(r"(\d{1,2})[:\uff1a](\d{2})\s*(?:的时候|时)"),
]


class Reminder:
    def __init__(self, content: str, trigger_time: float, reminder_id: str = None,
                 created_at: float = None, triggered: bool = False):
        self.id = reminder_id or f"r_{int(time.time() * 1000)}"
        self.content = content
        self.trigger_time = trigger_time
        self.created_at = created_at or time.time()
        self.triggered = triggered

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "trigger_time": self.trigger_time,
            "trigger_time_readable": datetime.fromtimestamp(self.trigger_time).strftime("%Y-%m-%d %H:%M"),
            "created_at": self.created_at,
            "triggered": self.triggered,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Reminder":
        return cls(
            content=d["content"],
            trigger_time=d["trigger_time"],
            reminder_id=d.get("id"),
            created_at=d.get("created_at"),
            triggered=d.get("triggered", False),
        )

    def is_due(self) -> bool:
        return not self.triggered and time.time() >= self.trigger_time


class ReminderScheduler:
    """Manages persistent reminders with background checking."""

    def __init__(self, data_dir: str = "."):
        self._data_dir = data_dir
        self._file_path = os.path.join(data_dir, REMINDERS_FILE)
        self._reminders: List[Reminder] = []
        self._task: Optional[asyncio.Task] = None
        self._on_reminder_callback: Optional[Callable] = None
        self._load()
        logger.info(f"ReminderScheduler: loaded {len(self._reminders)} reminders")

    def _load(self):
        if os.path.exists(self._file_path):
            try:
                with open(self._file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._reminders = [Reminder.from_dict(d) for d in data]
            except Exception as e:
                logger.warning(f"Failed to load reminders: {e}")
                self._reminders = []

    def _save(self):
        try:
            with open(self._file_path, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in self._reminders], f,
                          ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save reminders: {e}")

    def add_reminder(self, content: str, trigger_time: float) -> Reminder:
        reminder = Reminder(content=content, trigger_time=trigger_time)
        self._reminders.append(reminder)
        self._save()
        readable = datetime.fromtimestamp(trigger_time).strftime("%Y-%m-%d %H:%M")
        logger.info(f"Reminder added: '{content}' at {readable}")
        return reminder

    def get_pending_reminders(self) -> List[Reminder]:
        return [r for r in self._reminders if not r.triggered]

    def get_due_reminders(self) -> List[Reminder]:
        return [r for r in self._reminders if r.is_due()]

    def mark_triggered(self, reminder_id: str):
        for r in self._reminders:
            if r.id == reminder_id:
                r.triggered = True
                self._save()
                return

    def remove_old_triggered(self, max_age_days: int = 7):
        cutoff = time.time() - max_age_days * 86400
        self._reminders = [
            r for r in self._reminders
            if not (r.triggered and r.created_at < cutoff)
        ]
        self._save()

    def start(self, callback: Callable):
        """Start the background checker. callback(reminder) is called for due reminders."""
        self._on_reminder_callback = callback
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._check_loop())
            logger.info("ReminderScheduler: background checker started")

    def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            logger.info("ReminderScheduler: background checker stopped")

    async def _check_loop(self):
        """Check for due reminders every 3 seconds."""
        try:
            while True:
                await asyncio.sleep(3)
                due = self.get_due_reminders()
                for reminder in due:
                    self.mark_triggered(reminder.id)
                    if self._on_reminder_callback:
                        try:
                            await self._on_reminder_callback(reminder)
                        except Exception as e:
                            logger.error(f"Reminder callback error: {e}")
                self.remove_old_triggered()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"ReminderScheduler check loop error: {e}")


class IdleChatScheduler:
    """Proactively initiates conversations based on time-of-day rules.

    Frequency rules:
    - 08:00-21:00  normal hours   → random interval 15-40 min
    - 21:00-01:00  quiet hours    → random interval 50-90 min (less frequent)
    - 01:00-08:00  sleep hours    → no proactive chat
    """

    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self._on_idle_chat_callback: Optional[Callable] = None
        self._last_interaction: float = time.time()
        self._last_proactive: float = 0
        self._min_silence_before_chat: int = 180  # at least 3 min silence before speaking up

    def record_interaction(self):
        """Call this whenever user or bot speaks, to reset the idle timer."""
        self._last_interaction = time.time()

    def start(self, callback: Callable):
        self._on_idle_chat_callback = callback
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._idle_loop())
            logger.info("IdleChatScheduler: started")

    def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()

    def _get_time_config(self) -> Optional[Dict]:
        """Returns interval config for the current hour, or None if sleep hours."""
        hour = datetime.now().hour
        if 8 <= hour < 21:
            return {"min_interval": 900, "max_interval": 2400}  # 15-40 min
        elif 21 <= hour or hour < 1:
            return {"min_interval": 3000, "max_interval": 5400}  # 50-90 min
        else:
            return None  # 01:00-08:00 sleep, no proactive chat

    async def _idle_loop(self):
        import random
        try:
            next_chat_delay = random.randint(900, 2400)
            while True:
                await asyncio.sleep(10)

                config = self._get_time_config()
                if config is None:
                    continue

                idle_seconds = time.time() - self._last_interaction
                since_last_proactive = time.time() - self._last_proactive

                if (idle_seconds >= self._min_silence_before_chat
                        and since_last_proactive >= next_chat_delay):
                    self._last_proactive = time.time()
                    self._last_interaction = time.time()
                    next_chat_delay = random.randint(
                        config["min_interval"], config["max_interval"]
                    )
                    if self._on_idle_chat_callback:
                        try:
                            await self._on_idle_chat_callback()
                        except Exception as e:
                            logger.error(f"IdleChatScheduler callback error: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"IdleChatScheduler loop error: {e}")


def get_sleepy_state() -> Optional[str]:
    """Returns a sleepy state description if it's late, or None."""
    hour = datetime.now().hour
    if 1 <= hour < 8:
        return "sleeping"
    elif 21 <= hour or hour < 1:
        return "drowsy"
    return None


def parse_reminder_time(text: str) -> Optional[float]:
    """Parse a time expression from user text, return Unix timestamp or None."""
    now = datetime.now()

    for pattern, unit in RELATIVE_TIME_PATTERNS:
        match = pattern.search(text)
        if match:
            if unit == "half_hour":
                target = now + timedelta(minutes=30)
                return target.timestamp()
            value = int(match.group(1))
            if unit == "minutes":
                target = now + timedelta(minutes=value)
            elif unit == "hours":
                target = now + timedelta(hours=value)
            elif unit == "seconds":
                target = now + timedelta(seconds=value)
            else:
                continue
            return target.timestamp()

    is_tomorrow = "明天" in text or "明日" in text
    is_afternoon = "下午" in text or "晚上" in text

    for pattern in ABSOLUTE_TIME_PATTERNS:
        match = pattern.search(text)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2)) if match.group(2) else 0

            if is_afternoon and hour < 12:
                hour += 12

            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if is_tomorrow:
                target += timedelta(days=1)
            elif target <= now:
                target += timedelta(days=1)

            return target.timestamp()

    return None


def detect_reminder_intent(text: str) -> Optional[Dict[str, Any]]:
    """Detect if user wants to set a reminder. Returns {content, trigger_time} or None."""
    reminder_keywords = [
        re.compile(r"(?:提醒|叫)我(.+?)(?:去|做|要|把)?(.+)", re.DOTALL),
        re.compile(r"(.+?)(?:的时候|时|后)(?:提醒|叫|告诉)我(.+)", re.DOTALL),
        re.compile(r"(?:提醒|remind).*?(?:我|me)\s*(.+)", re.IGNORECASE | re.DOTALL),
        re.compile(r"(\d+\s*(?:分钟|小时|秒)[后以])\s*(?:提醒|叫|告诉)我(.+)", re.DOTALL),
    ]

    for pattern in reminder_keywords:
        match = pattern.search(text)
        if match:
            trigger_time = parse_reminder_time(text)
            if trigger_time:
                content = text.strip()
                return {"content": content, "trigger_time": trigger_time}

    if parse_reminder_time(text) and any(kw in text for kw in ["提醒", "叫我", "告诉我", "remind"]):
        trigger_time = parse_reminder_time(text)
        return {"content": text.strip(), "trigger_time": trigger_time}

    return None
