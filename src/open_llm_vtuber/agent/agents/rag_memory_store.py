"""
RAG Memory Store - ChromaDB-based long-term memory with 2-week expiry.

Stores conversation memories as vector embeddings for semantic retrieval.
Automatically prunes entries older than the configured retention period.
"""

import time
import os
import hashlib
from typing import List, Dict, Optional
from loguru import logger

import chromadb
from chromadb.config import Settings


RETENTION_SECONDS = 14 * 24 * 3600  # 2 weeks


class RAGMemoryStore:
    """Persistent vector memory with automatic expiration."""

    def __init__(
        self,
        db_path: str = "./rag_memory_db",
        collection_name: str = "waifu_memory",
        retention_days: int = 14,
        max_results: int = 8,
    ):
        self._retention_seconds = retention_days * 24 * 3600
        self._max_results = max_results
        self._db_path = os.path.abspath(db_path)

        os.makedirs(self._db_path, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self._db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"RAGMemoryStore initialized: {self._collection.count()} memories in DB, "
            f"retention={retention_days}d, path={self._db_path}"
        )
        self._cleanup_expired()

    def _make_id(self, content: str, timestamp: float) -> str:
        raw = f"{content}:{timestamp}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def add_memory(self, content: str, role: str = "system", importance: str = "normal"):
        """Store a memory entry with timestamp metadata."""
        if not content or not content.strip():
            return
        ts = time.time()
        doc_id = self._make_id(content, ts)

        self._collection.upsert(
            ids=[doc_id],
            documents=[content],
            metadatas=[{
                "role": role,
                "importance": importance,
                "timestamp": ts,
                "readable_time": time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)),
            }],
        )
        logger.debug(f"RAG memory added [{importance}]: {content[:80]}...")

    def add_conversation_summary(self, user_msg: str, assistant_msg: str):
        """Store a conversation turn as a combined memory for better retrieval."""
        combined = f"用户说: {user_msg}\n回复: {assistant_msg}"
        self.add_memory(combined, role="conversation", importance="normal")

    def query(self, query_text: str, n_results: Optional[int] = None) -> List[Dict]:
        """Retrieve relevant memories via semantic search."""
        self._cleanup_expired()

        n = n_results or self._max_results
        count = self._collection.count()
        if count == 0:
            return []

        n = min(n, count)
        results = self._collection.query(
            query_texts=[query_text],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        memories = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                memories.append({
                    "content": doc,
                    "role": meta.get("role", "unknown"),
                    "importance": meta.get("importance", "normal"),
                    "time": meta.get("readable_time", "unknown"),
                    "relevance": round(1 - dist, 3),
                })

        memories.sort(key=lambda m: m["relevance"], reverse=True)
        return memories

    def _cleanup_expired(self):
        """Remove memories older than retention period."""
        cutoff = time.time() - self._retention_seconds
        try:
            results = self._collection.get(
                include=["metadatas"],
            )
            if not results or not results["ids"]:
                return

            expired_ids = []
            for doc_id, meta in zip(results["ids"], results["metadatas"]):
                ts = meta.get("timestamp", 0)
                if ts < cutoff:
                    expired_ids.append(doc_id)

            if expired_ids:
                self._collection.delete(ids=expired_ids)
                logger.info(f"RAG memory cleanup: removed {len(expired_ids)} expired entries")
        except Exception as e:
            logger.warning(f"RAG memory cleanup error: {e}")

    def get_stats(self) -> Dict:
        return {
            "total_memories": self._collection.count(),
            "db_path": self._db_path,
            "retention_days": self._retention_seconds // (24 * 3600),
        }
