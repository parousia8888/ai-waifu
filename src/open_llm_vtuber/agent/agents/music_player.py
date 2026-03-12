"""
Music Player - Search via pyncm (NetEase) for metadata, play via yt-dlp + afplay.
Uses pyncm for fast song title lookup, yt-dlp to fetch audio from YouTube,
and macOS afplay for background playback.
"""

import subprocess
import os
import re
import signal
from typing import Optional, Dict
from loguru import logger

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
YTDLP_PATH = os.path.join(_PROJECT_ROOT, ".venv", "bin", "yt-dlp")
FFMPEG_PATH = os.path.expanduser("~/.local/bin/ffmpeg")
TEMP_AUDIO = "/tmp/ai_waifu_music.m4a"


class MusicPlayer:
    def __init__(self, service: str = "netease"):
        self.service = service
        self._bg_process: Optional[subprocess.Popen] = None

    def search_and_play(self, query: str) -> str:
        song_name, artist = self._search_metadata(query)
        display = f"{song_name} - {artist}" if artist else song_name

        yt_query = f"{song_name} {artist}" if artist else query
        self._play_from_youtube(yt_query)

        return f"正在播放: {display}"

    def _search_metadata(self, query: str) -> tuple:
        """Use pyncm to search NetEase for song name and artist."""
        try:
            from pyncm import apis
            result = apis.cloudsearch.GetSearchResult(query, limit=1)
            songs = result.get("result", {}).get("songs", [])
            if songs:
                song = songs[0]
                name = song.get("name", query)
                artists = song.get("ar", [])
                artist = artists[0]["name"] if artists else ""
                return (name, artist)
        except Exception as e:
            logger.warning(f"pyncm search failed: {e}")
        return (query, "")

    def _play_from_youtube(self, query: str):
        """Download audio from YouTube and play with afplay in background."""
        self._stop_current()

        try:
            os.remove(TEMP_AUDIO)
        except OSError:
            pass

        safe_query = query.replace('"', '\\"').replace("'", "\\'")
        temp_template = TEMP_AUDIO.replace(".m4a", ".%(ext)s")
        cmd = (
            f'"{YTDLP_PATH}" -x --audio-format m4a '
            f'--ffmpeg-location "{FFMPEG_PATH}" '
            f'-o "{temp_template}" "ytsearch1:{safe_query}" '
            f'--no-playlist --quiet --no-warnings 2>/dev/null && '
            f'afplay "{TEMP_AUDIO}"'
        )
        self._bg_process = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        logger.info(f"YouTube playback started in background (pid={self._bg_process.pid})")

    def _stop_current(self):
        subprocess.run(
            ["pkill", "-f", f"afplay {TEMP_AUDIO}"],
            capture_output=True, timeout=3,
        )
        if self._bg_process and self._bg_process.poll() is None:
            try:
                os.killpg(os.getpgid(self._bg_process.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
        self._bg_process = None

    def control_playback(self, action: str) -> str:
        if action == "pause":
            subprocess.run(
                ["pkill", "-STOP", "-f", f"afplay {TEMP_AUDIO}"],
                capture_output=True, timeout=3,
            )
            return "已暂停播放"
        elif action == "resume":
            subprocess.run(
                ["pkill", "-CONT", "-f", f"afplay {TEMP_AUDIO}"],
                capture_output=True, timeout=3,
            )
            return "继续播放"
        elif action == "next":
            return "暂不支持切歌，请告诉我你想听什么"
        elif action == "stop":
            self._stop_current()
            return "已停止播放"
        return "未知操作"

    def set_volume(self, level: int) -> str:
        level = max(0, min(100, level))
        script = f'set volume output volume {level}'
        try:
            subprocess.run(["osascript", "-e", script], timeout=5)
        except Exception as e:
            logger.error(f"Volume error: {e}")
        return f"音量已调至 {level}"


# --- Intent Detection ---

QUESTION_CONTEXT = re.compile(
    r"(?:在哪|哪里|怎么|什么时候|为什么|能不能|是不是|有没有).{0,6}播放"
)

MUSIC_PLAY_PATTERNS = [
    re.compile(r"^播放[：:\s]*(.{2,})", re.IGNORECASE),
    re.compile(r"(?:我想听|给我放|帮我放|帮我播放)\s+(.{2,})", re.IGNORECASE),
    re.compile(r"(?:放一首|来一首|唱一首|来首|放首)\s*(.{2,})", re.IGNORECASE),
    re.compile(r"play\s+(.+)", re.IGNORECASE),
]

MUSIC_CONTROL_PATTERNS = {
    "pause": re.compile(
        r"(?:暂停|停一下|别放了|停下来|不想听了|不听了|别播了|stop|pause)",
        re.IGNORECASE,
    ),
    "resume": re.compile(
        r"(?:继续播放|接着放|接着听|继续放|resume)",
        re.IGNORECASE,
    ),
    "stop": re.compile(
        r"(?:停止播放|关掉音乐|把音乐关了|stop music)",
        re.IGNORECASE,
    ),
}

VOLUME_PATTERN = re.compile(
    r"(?:音量|声音)(?:调到|设为|设置为?|改为)?\s*(\d+)", re.IGNORECASE,
)

TRAILING_JUNK = re.compile(r"[。！!？?，,、\s\[\]【】]+$")
TRAILING_SONG_SUFFIX = re.compile(
    r"(?:的歌|的曲子|的音乐|这首歌|那首歌|吧|啊|呀|嘛|好不好|好吗|可以吗)$"
)
LEADING_JUNK = re.compile(r"^(?:一首|一下|一个|那个|那首)\s*")


def _clean_query(raw: str) -> str:
    q = raw.strip()
    q = TRAILING_JUNK.sub("", q)
    for _ in range(3):
        q = TRAILING_SONG_SUFFIX.sub("", q)
    q = TRAILING_JUNK.sub("", q)
    q = LEADING_JUNK.sub("", q)
    return q.strip()


def detect_music_intent(text: str) -> Optional[Dict]:
    """
    Detect music intent from user text.
    Returns dict with 'action' and optional 'query' / 'level', or None.
    """
    clean = text.strip().rstrip("。！!？?，, ")

    if QUESTION_CONTEXT.search(clean):
        return None

    for action, pattern in MUSIC_CONTROL_PATTERNS.items():
        if pattern.search(clean):
            return {"action": action}

    vol_match = VOLUME_PATTERN.search(clean)
    if vol_match:
        return {"action": "volume", "level": int(vol_match.group(1))}

    for pattern in MUSIC_PLAY_PATTERNS:
        match = pattern.search(clean)
        if match:
            query = _clean_query(match.group(1))
            if query and len(query) >= 2:
                return {"action": "play", "query": query}

    return None
