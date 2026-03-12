"""
File Reader - Reads local files for the RAG agent to discuss with user.

Supports text files, code files, PDFs (text extraction), and common formats.
"""

import os
import mimetypes
from typing import Optional
from loguru import logger


MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_CHARS = 20000

TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp",
    ".h", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt",
    ".json", ".yaml", ".yml", ".toml", ".xml", ".html", ".css", ".scss",
    ".sh", ".bash", ".zsh", ".bat", ".ps1",
    ".sql", ".r", ".m", ".lua", ".pl", ".scala",
    ".csv", ".log", ".ini", ".cfg", ".conf", ".env",
    ".tex", ".rst", ".org", ".adoc",
}


def read_file(file_path: str, max_chars: int = MAX_CHARS) -> Optional[str]:
    """Read a file and return its content as a string.

    Returns None if the file can't be read.
    """
    path = os.path.expanduser(file_path)
    if not os.path.isfile(path):
        logger.warning(f"File not found: {path}")
        return None

    size = os.path.getsize(path)
    if size > MAX_FILE_SIZE:
        logger.warning(f"File too large ({size} bytes): {path}")
        return None

    ext = os.path.splitext(path)[1].lower()

    if ext in TEXT_EXTENSIONS:
        return _read_text_file(path, max_chars)

    mime, _ = mimetypes.guess_type(path)
    if mime and mime.startswith("text/"):
        return _read_text_file(path, max_chars)

    if ext == ".pdf":
        return _read_pdf(path, max_chars)

    return _try_read_as_text(path, max_chars)


def _read_text_file(path: str, max_chars: int) -> Optional[str]:
    encodings = ["utf-8", "gbk", "gb2312", "shift_jis", "latin-1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                content = f.read(max_chars)
            if len(content) >= max_chars:
                content += "\n...(文件内容过长，已截断)"
            return content
        except (UnicodeDecodeError, UnicodeError):
            continue
    logger.warning(f"Could not decode file with any encoding: {path}")
    return None


def _read_pdf(path: str, max_chars: int) -> Optional[str]:
    try:
        import subprocess
        result = subprocess.run(
            ["python3", "-c", f"""
import sys
try:
    from PyPDF2 import PdfReader
    reader = PdfReader("{path}")
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        if len(text) > {max_chars}:
            break
    print(text[:{max_chars}])
except ImportError:
    print("[PDF reading requires PyPDF2, not installed]")
"""],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception as e:
        logger.warning(f"PDF read error: {e}")
    return f"[无法读取PDF文件: {path}]"


def _try_read_as_text(path: str, max_chars: int) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read(max_chars)
        return content
    except Exception:
        return f"[不支持的文件格式: {os.path.basename(path)}]"


def list_directory(dir_path: str, max_items: int = 50) -> Optional[str]:
    """List files in a directory."""
    path = os.path.expanduser(dir_path)
    if not os.path.isdir(path):
        return None

    items = []
    try:
        for i, entry in enumerate(sorted(os.listdir(path))):
            if i >= max_items:
                items.append(f"...还有更多文件 (共 {len(os.listdir(path))} 项)")
                break
            full = os.path.join(path, entry)
            if os.path.isdir(full):
                items.append(f"📁 {entry}/")
            else:
                size = os.path.getsize(full)
                items.append(f"📄 {entry} ({_human_size(size)})")
    except PermissionError:
        return f"[没有权限访问: {path}]"

    return "\n".join(items)


def _human_size(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.0f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"
