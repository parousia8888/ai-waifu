"""
RAG Memory Agent - Extends BasicMemoryAgent with:
- Long-term semantic memory via ChromaDB (2-week retention)
- Memory save/recall triggered by conversation context
- Local file reading capability
- Reduced token usage via RAG retrieval instead of full history
- Proactive reminder scheduling
"""

import os
import re
from datetime import datetime
from typing import (
    AsyncIterator,
    List,
    Dict,
    Any,
    Union,
    Optional,
    Literal,
)
from loguru import logger

from .basic_memory_agent import BasicMemoryAgent
from .rag_memory_store import RAGMemoryStore
from . import file_reader
from ..output_types import SentenceOutput
from ..stateless_llm.stateless_llm_interface import StatelessLLMInterface
from ...config_manager import TTSPreprocessorConfig
from ..input_types import BatchInput, TextSource
from ...mcpp.tool_manager import ToolManager
from ...mcpp.tool_executor import ToolExecutor
from ...reminder_scheduler import detect_reminder_intent, ReminderScheduler, get_sleepy_state
from .self_notes import SelfNotes
from .music_player import MusicPlayer, detect_music_intent, _clean_query

_global_reminder_scheduler: Optional[ReminderScheduler] = None


def set_global_reminder_scheduler(scheduler: ReminderScheduler):
    global _global_reminder_scheduler
    _global_reminder_scheduler = scheduler


REMEMBER_PATTERNS = [
    re.compile(r"记住(.+)", re.IGNORECASE),
    re.compile(r"记一下(.+)", re.IGNORECASE),
    re.compile(r"别忘了(.+)", re.IGNORECASE),
    re.compile(r"remember\s+(?:that\s+)?(.+)", re.IGNORECASE),
    re.compile(r"请?记住(.+)", re.IGNORECASE),
    re.compile(r"帮我记(.+)", re.IGNORECASE),
    re.compile(r"我告诉你(.+)", re.IGNORECASE),
]

FILE_PATTERNS = [
    re.compile(r"(?:读取?|打开|查看|看看|read)\s*(?:文件|file)?\s*[：:]*\s*(.+)", re.IGNORECASE),
    re.compile(r"(?:文件|file)\s*[：:]*\s*(.+)", re.IGNORECASE),
    re.compile(r"(?:列出|list)\s*(?:目录|directory|文件夹|folder)\s*[：:]*\s*(.+)", re.IGNORECASE),
]

SELF_NOTE_PATTERNS = [
    re.compile(r"(?:^|[，,。\s])以后(?:要|都|得|请你|你要|你得)(.+)", re.IGNORECASE),
    re.compile(r"你(?:以后|之后|今后)(?:要|都|得)(.+)", re.IGNORECASE),
    re.compile(r"(?:不要|别|不许|不准)再(.+)", re.IGNORECASE),
    re.compile(r"(?:改成|换成|调整为)(.+)", re.IGNORECASE),
]

MAX_SHORT_MEMORY = 10


class RAGMemoryAgent(BasicMemoryAgent):
    """Agent with RAG-based long-term memory and file access."""

    def __init__(
        self,
        llm: StatelessLLMInterface,
        system: str,
        live2d_model,
        tts_preprocessor_config: TTSPreprocessorConfig = None,
        faster_first_response: bool = True,
        segment_method: str = "pysbd",
        use_mcpp: bool = False,
        interrupt_method: Literal["system", "user"] = "user",
        tool_prompts: Dict[str, str] = None,
        tool_manager: Optional[ToolManager] = None,
        tool_executor: Optional[ToolExecutor] = None,
        mcp_prompt_string: str = "",
        rag_db_path: str = "./rag_memory_db",
        rag_retention_days: int = 14,
        rag_max_results: int = 8,
        max_short_memory: int = 10,
        music_service: str = "netease",
    ):
        super().__init__(
            llm=llm,
            system=system,
            live2d_model=live2d_model,
            tts_preprocessor_config=tts_preprocessor_config,
            faster_first_response=faster_first_response,
            segment_method=segment_method,
            use_mcpp=use_mcpp,
            interrupt_method=interrupt_method,
            tool_prompts=tool_prompts,
            tool_manager=tool_manager,
            tool_executor=tool_executor,
            mcp_prompt_string=mcp_prompt_string,
        )

        self._rag_store = RAGMemoryStore(
            db_path=rag_db_path,
            retention_days=rag_retention_days,
            max_results=rag_max_results,
        )
        self._max_short_memory = max_short_memory
        self._original_system = system
        self._self_notes = SelfNotes(data_dir=rag_db_path if os.path.isdir(rag_db_path) else ".")
        self._music_player = MusicPlayer(service=music_service)

        logger.info(
            f"RAGMemoryAgent initialized. "
            f"DB: {rag_db_path}, retention: {rag_retention_days}d, "
            f"short_memory: {max_short_memory} turns, "
            f"music: {music_service}, "
            f"self_notes: {len(self._self_notes.get_all_notes())}"
        )

    def _detect_remember_intent(self, text: str) -> Optional[str]:
        """Detect if user wants to store a memory. Returns memory content or None."""
        for pattern in REMEMBER_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return None

    def _detect_file_request(self, text: str) -> Optional[str]:
        """Detect if user wants to read a file. Returns file path or None."""
        for pattern in FILE_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip().strip('"').strip("'")
        return None

    def _trim_working_memory(self):
        """Keep working memory short to reduce token usage."""
        if len(self._memory) > self._max_short_memory * 2:
            overflow = self._memory[:-self._max_short_memory * 2]
            for i in range(0, len(overflow) - 1, 2):
                user_msg = overflow[i].get("content", "") if overflow[i].get("role") == "user" else ""
                asst_msg = overflow[i + 1].get("content", "") if i + 1 < len(overflow) and overflow[i + 1].get("role") == "assistant" else ""
                if user_msg and asst_msg:
                    self._rag_store.add_conversation_summary(user_msg, asst_msg)
            self._memory = self._memory[-self._max_short_memory * 2:]

    def _build_rag_context(self, query: str) -> str:
        """Build context from RAG retrieval results."""
        memories = self._rag_store.query(query)
        if not memories:
            return ""

        lines = ["[长期记忆中检索到的相关内容]"]
        for m in memories:
            tag = ""
            if m["importance"] == "important":
                tag = "[重要] "
            lines.append(f"- ({m['time']}, 相关度{m['relevance']}) {tag}{m['content']}")
        lines.append("[长期记忆结束]")
        return "\n".join(lines)

    def _detect_self_note(self, text: str) -> bool:
        """Detect if user is giving a behavioral instruction and save it."""
        music_keywords = ["播放", "放一首", "来一首", "我想听", "帮我放"]
        if any(kw in text for kw in music_keywords):
            return False

        behavioral_keywords = [
            "以后要", "以后都", "以后得", "以后请", "以后你",
            "之后要", "之后都", "今后",
            "不要再", "别再", "不许再", "不准再",
            "改成", "换成", "调整为",
        ]
        if not any(kw in text for kw in behavioral_keywords):
            return False

        for pattern in SELF_NOTE_PATTERNS:
            match = pattern.search(text)
            if match:
                note_content = text.strip()
                if len(note_content) < 5 or len(note_content) > 200:
                    return False
                result = self._self_notes.add_note(note_content, category="user_instruction")
                if result == "added":
                    logger.info(f"Self-note saved: {note_content[:60]}")
                    return True
        return False

    def _handle_music_intent(self, text: str) -> Optional[str]:
        """Detect music intent and execute it. Returns status string or None."""
        intent = detect_music_intent(text)
        if not intent:
            return None

        action = intent["action"]
        if action == "play":
            result = self._music_player.search_and_play(intent["query"])
            logger.info(f"Music play: {intent['query']} -> {result}")
            return f"[音乐操作] {result}"
        elif action == "volume":
            result = self._music_player.set_volume(intent["level"])
            return f"[音乐操作] {result}"
        else:
            result = self._music_player.control_playback(action)
            return f"[音乐操作] {result}"

    def _detect_and_create_reminder(self, text: str) -> Optional[str]:
        """Detect reminder intent and create a scheduled reminder. Returns info string or None."""
        global _global_reminder_scheduler
        if not _global_reminder_scheduler:
            return None

        result = detect_reminder_intent(text)
        if result:
            reminder = _global_reminder_scheduler.add_reminder(
                content=result["content"],
                trigger_time=result["trigger_time"],
            )
            trigger_dt = datetime.fromtimestamp(result["trigger_time"])
            time_str = trigger_dt.strftime("%H:%M")
            self._rag_store.add_memory(
                f"设置了提醒: {result['content']} (时间: {time_str})",
                role="reminder", importance="important",
            )
            return f"[已设置提醒，将在 {time_str} 触发]"
        return None

    def _to_messages(self, input_data: BatchInput) -> List[Dict[str, Any]]:
        """Override: inject RAG context and handle special commands."""
        text_prompt = self._to_text_prompt(input_data)

        memory_to_store = self._detect_remember_intent(text_prompt)
        if memory_to_store:
            self._rag_store.add_memory(memory_to_store, role="user_explicit", importance="important")
            logger.info(f"Explicit memory stored: {memory_to_store[:60]}...")

        self._detect_self_note(text_prompt)

        reminder_info = self._detect_and_create_reminder(text_prompt)

        file_path = self._detect_file_request(text_prompt)
        file_content = None
        if file_path:
            if file_path.rstrip("/").endswith(("/", "\\")) or not "." in file_path.split("/")[-1]:
                dir_listing = file_reader.list_directory(file_path)
                if dir_listing:
                    file_content = f"[目录 {file_path} 的内容]\n{dir_listing}"
            else:
                content = file_reader.read_file(file_path)
                if content:
                    file_content = f"[文件 {file_path} 的内容]\n{content}"
                else:
                    file_content = f"[无法读取文件: {file_path}]"

        music_info = self._handle_music_intent(text_prompt)

        rag_context = self._build_rag_context(text_prompt) if text_prompt else ""

        augmented_system = self._original_system

        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]
        augmented_system = f"{augmented_system}\n\n[当前时间: {now_str} {weekday}]"

        sleepy = get_sleepy_state()
        if sleepy == "sleeping":
            augmented_system = f"{augmented_system}\n[你正在睡觉，被用户吵醒了。回复要迷糊、简短、带起床气。]"
        elif sleepy == "drowsy":
            augmented_system = f"{augmented_system}\n[你现在有点困了，说话慵懒简短一些。]"

        notes_section = self._self_notes.build_prompt_section()
        if notes_section:
            augmented_system = f"{augmented_system}\n\n{notes_section}"
        if rag_context:
            augmented_system = f"{augmented_system}\n\n{rag_context}"
        if file_content:
            augmented_system = f"{augmented_system}\n\n{file_content}"
        if reminder_info:
            augmented_system = f"{augmented_system}\n\n{reminder_info}"
        if music_info:
            augmented_system = f"{augmented_system}\n\n{music_info}"
        self.set_system(augmented_system)

        self._trim_working_memory()

        messages = self._memory.copy()
        user_content = []
        if text_prompt:
            user_content.append({"type": "text", "text": text_prompt})

        if input_data.images:
            for img_data in input_data.images:
                if isinstance(img_data.data, str) and img_data.data.startswith("data:image"):
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": img_data.data, "detail": "auto"},
                    })

        if user_content:
            messages.append({"role": "user", "content": user_content})
            skip_memory = False
            if input_data.metadata and input_data.metadata.get("skip_memory", False):
                skip_memory = True
            if not skip_memory:
                self._add_message(
                    text_prompt if text_prompt else "[User provided image(s)]", "user"
                )

        return messages

    _AI_PLAY_PATTERN = re.compile(
        r"(?:^|[\n。！!])播放[：:\s]*([^\n。！!？?\"\"\"、,，]{2,30})(?:[。！!？?\n\[]|$)"
    )

    def _check_ai_music_command(self, text: str):
        """If the AI's response contains '播放xxx', trigger actual playback."""
        match = self._AI_PLAY_PATTERN.search(text)
        if match:
            query = _clean_query(match.group(1))
            if query and len(query) >= 2:
                result = self._music_player.search_and_play(query)
                logger.info(f"AI-triggered music play: {query} -> {result}")

    def _add_message(self, message, role, display_text=None, skip_memory=False):
        """Override: also store assistant responses to RAG for long-term memory."""
        super()._add_message(message, role, display_text, skip_memory)

        if skip_memory:
            return
        if role == "assistant" and isinstance(message, str):
            self._check_ai_music_command(message)
            if len(message) > 10 and len(self._memory) >= 2:
                prev = self._memory[-2] if len(self._memory) >= 2 else None
                if prev and prev.get("role") == "user":
                    self._rag_store.add_conversation_summary(
                        prev.get("content", ""), message
                    )
