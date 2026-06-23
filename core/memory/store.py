from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from core.message import Message

logger = logging.getLogger(__name__)


MEMORY_FACT = "fact"
MEMORY_SUMMARY = "summary"
MEMORY_DECISION = "decision"
MEMORY_FILE_CHANGE = "file_change"
MEMORY_TEST_RESULT = "test_result"
MEMORY_ERROR = "error"
MEMORY_USER_PREFERENCE = "user_preference"


class ShortMemory:

    def __init__(self, max_length: int = 20, max_tokens: int = 4000):
        """
        Args:
            max_length: 最多保留的消息条数
            max_tokens: 最多保留的 token 数（超过时从最早开始丢弃）
        """
        self.max_length = max_length
        self.max_tokens = max_tokens
        self.messages: List[Message] = []

    def add_message(self, message: Message):
        """添加消息，自动按数量和 token 数截断"""
        self.messages.append(message)
        self._trim()

    def replace_messages(self, messages: List[Message]):
        """整体替换短期记忆，并重新应用长度/token 限制"""
        self.messages = list(messages)
        self._trim()

    def _trim(self):
        """按消息数和 token 数裁剪最早消息"""
        while len(self.messages) > self.max_length:
            self.messages.pop(0)

        while self.total_tokens() > self.max_tokens and len(self.messages) > 1:
            self.messages.pop(0)

    def get_recent_messages(self, n: Optional[int] = None) -> List[Message]:
        """获取最近的 n 条消息"""
        if n is None or n > len(self.messages):
            return self.messages[:]
        return self.messages[-n:]

    def get_last_user_message(self) -> Optional[Message]:
        """获取最后一条用户消息"""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg
        return None

    def total_tokens(self) -> int:
        """计算所有消息的总 token 数"""
        return sum(msg.token_count() for msg in self.messages)

    def clear(self):
        """清空短期记忆"""
        self.messages.clear()

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """序列化为字典列表"""
        return [msg.to_dict() for msg in self.messages]

    @classmethod
    def from_dict_list(cls, data: List[Dict[str, Any]],
                       max_length: int = 20, max_tokens: int = 4000) -> "ShortMemory":
        """从字典列表反序列化"""
        mem = cls(max_length=max_length, max_tokens=max_tokens)
        mem.messages = [Message.from_dict(d) for d in data]
        return mem


@dataclass
class MemoryItem:
    """A structured long-term memory record."""

    content: str
    type: str = MEMORY_FACT
    tags: List[str] = field(default_factory=list)
    source: Optional[str] = None
    importance: float = 0.5
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = field(default_factory=lambda: datetime.now().timestamp())
    embedding: Optional[Any] = None
    content_hash: Optional[str] = None
    superseded_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.content = str(self.content).strip()
        self.type = self.type or MEMORY_FACT
        self.tags = sorted({tag.strip() for tag in self.tags if tag and tag.strip()})
        self.importance = _clamp_float(self.importance, 0.0, 1.0)
        if self.embedding is None:
            self.embedding = _text_vector(self.content)
        if self.content_hash is None:
            self.content_hash = _content_hash(self.content)

    @property
    def active(self) -> bool:
        return self.superseded_by is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "tags": self.tags,
            "source": self.source,
            "importance": self.importance,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "embedding": self.embedding,
            "content_hash": self.content_hash,
            "superseded_by": self.superseded_by,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        return cls(
            id=data.get("id") or uuid4().hex,
            type=data.get("type") or MEMORY_FACT,
            content=data.get("content", ""),
            tags=list(data.get("tags", [])),
            source=data.get("source"),
            importance=data.get("importance", 0.5),
            created_at=data.get("created_at", datetime.now().timestamp()),
            updated_at=data.get("updated_at", datetime.now().timestamp()),
            embedding=data.get("embedding"),
            content_hash=data.get("content_hash"),
            superseded_by=data.get("superseded_by"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MemorySearchResult:
    item: MemoryItem
    score: float
    reason: str = ""


class MemorySummarizer:
    """Builds useful summaries from recent agent messages."""

    SECTION_RULES = (
        ("用户目标", ("用户请求", "需要", "实现", "修复", "修改", "优化", "测试")),
        ("工具调用", ("[调用工具", "调用工具")),
        ("文件与代码线索", ("read_file", "write_file", "edit_file", ".py", ".md", "文件")),
        ("错误与阻塞", ("[错误]", "错误", "失败", "Traceback", "Exception", "权限拒绝")),
        ("测试与验证", ("PASS", "通过", "pytest", "compileall", "测试", "验证")),
        ("助手结论", ("完成", "结论", "总结")),
    )

    def summarize(self, messages: List[Message]) -> str:
        if not messages:
            return "无可总结的对话。"

        sections: Dict[str, List[str]] = {name: [] for name, _ in self.SECTION_RULES}
        for msg in messages:
            line = self._message_line(msg)
            for section, keywords in self.SECTION_RULES:
                if self._matches(line, keywords):
                    self._append_unique(sections[section], line, limit=6)

        parts = []
        for section, lines in sections.items():
            if lines:
                parts.append(section + "：")
                parts.extend(f"- {line}" for line in lines)

        if not parts:
            compact = [self._message_line(m) for m in messages[-6:]]
            parts.append("对话摘要：")
            parts.extend(f"- {line}" for line in compact if line)

        return "\n".join(parts)

    @staticmethod
    def _message_line(msg: Message) -> str:
        name = f"/{msg.name}" if msg.name else ""
        content = _shorten(" ".join(str(msg.content).split()), 260)
        return f"{msg.role}{name}: {content}"

    @staticmethod
    def _matches(text: str, keywords: Iterable[str]) -> bool:
        lower = text.lower()
        return any(keyword.lower() in lower for keyword in keywords)

    @staticmethod
    def _append_unique(lines: List[str], line: str, limit: int) -> None:
        if line and line not in lines:
            lines.append(line)
        if len(lines) > limit:
            del lines[:-limit]


class LongMemory:
    """Structured long-term memory with retrieval and deduplication."""

    def __init__(
        self,
        storage_path: str = "memory_long.json",
        max_items: int = 300,
        similarity_threshold: float = 0.88,
        embedder: Optional[Any] = None,
    ):
        self.storage_path = storage_path
        self.max_items = max_items
        self.similarity_threshold = similarity_threshold
        self.embedder = embedder
        self.items: List[MemoryItem] = []
        self._summarizer = MemorySummarizer()
        self._load()

    @property
    def facts(self) -> List[str]:
        return [item.content for item in self.items if item.active and item.type == MEMORY_FACT]

    @property
    def summaries(self) -> List[str]:
        return [item.content for item in self.items if item.active and item.type == MEMORY_SUMMARY]

    def _load(self):
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            self.items = []
            return

        loaded = []
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            loaded = [MemoryItem.from_dict(item) for item in data.get("items", [])]
        elif isinstance(data, dict):
            for fact in data.get("facts", []):
                loaded.append(MemoryItem(content=fact, type=MEMORY_FACT, source="legacy"))
            for summary in data.get("summaries", []):
                loaded.append(MemoryItem(content=summary, type=MEMORY_SUMMARY, source="legacy"))

        self.items = [item for item in loaded if item.content]
        self._trim()

    def _save(self):
        """保存长期记忆到文件"""
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump({
                "version": 2,
                "items": [item.to_dict() for item in self.items],
                "facts": self.facts,
                "summaries": self.summaries,
            }, f, ensure_ascii=False, indent=2)

    def add_fact(
        self,
        fact: str,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        importance: float = 0.7,
    ) -> Optional[MemoryItem]:
        """添加一个关键事实"""
        return self.add_item(
            content=fact,
            memory_type=MEMORY_FACT,
            tags=tags,
            source=source,
            importance=importance,
        )

    def add_summary(
        self,
        summary: str,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        importance: float = 0.5,
    ) -> Optional[MemoryItem]:
        """添加一次对话总结"""
        return self.add_item(
            content=summary,
            memory_type=MEMORY_SUMMARY,
            tags=tags,
            source=source,
            importance=importance,
        )

    def add_item(
        self,
        content: str,
        memory_type: str = MEMORY_FACT,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[MemoryItem]:
        content = str(content or "").strip()
        if not content:
            return None

        item = MemoryItem(
            content=content,
            type=memory_type,
            tags=tags or _infer_tags(content, memory_type),
            source=source,
            importance=importance,
            metadata=metadata or {},
        )
        self._refresh_item_embedding(item)
        existing = self._find_duplicate(item)
        if existing is not None:
            self._merge_item(existing, item)
            self._save()
            return existing

        self.items.append(item)
        self._trim()
        self._save()
        return item

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_superseded: bool = False,
    ) -> List[MemorySearchResult]:
        query = str(query or "").strip()
        if not query:
            return []

        query_vec, query_kind = self._embed_text(query)
        results = []
        dirty = False
        for item in self.items:
            if not include_superseded and not item.active:
                continue
            if not self._matches_filters(item, filters):
                continue
            item_vec, item_dirty = self._ensure_item_embedding(item, preferred_kind=query_kind)
            dirty = dirty or item_dirty
            similarity = _cosine(query_vec, item_vec)
            overlap = self._term_overlap(_token_set(query), _token_set(item.content))
            recency = self._recency_score(item)
            score = (similarity * 0.68) + (overlap * 0.18) + (item.importance * 0.1) + (recency * 0.04)
            if score <= 0:
                continue
            results.append(MemorySearchResult(
                item=item,
                score=score,
                reason=f"similarity={similarity:.2f}, overlap={overlap:.2f}",
            ))

        results.sort(key=lambda result: result.score, reverse=True)
        if dirty:
            self._save()
        return results[:max(1, top_k)]

    def get_context(
        self,
        max_facts: int = 10,
        max_summaries: int = 3,
        query: Optional[str] = None,
        top_k: int = 6,
    ) -> str:
        """获取长期记忆上下文文本。传入 query 时优先返回相关记忆。"""
        if query:
            results = self.search(query, top_k=top_k)
            if results:
                lines = []
                for result in results:
                    item = result.item
                    tags = f" [{', '.join(item.tags)}]" if item.tags else ""
                    lines.append(f"- ({item.type}, score={result.score:.2f}){tags} {item.content}")
                return "【长期记忆 - 相关检索】\n" + "\n".join(lines)

        parts = []
        facts = self.facts
        summaries = self.summaries
        if facts:
            facts_text = "\n".join(f"- {f}" for f in facts[-max_facts:])
            parts.append(f"【长期记忆 - 关键事实】\n{facts_text}")
        if summaries:
            summaries_text = "\n".join(f"- {s}" for s in summaries[-max_summaries:])
            parts.append(f"【历史对话摘要】\n{summaries_text}")
        return "\n\n".join(parts)

    def summarize_and_store(self, recent_messages: List[Message],
                            llm_summarize_fn=None) -> Optional[str]:
        if not recent_messages:
            return None

        if llm_summarize_fn:
            summary = llm_summarize_fn(recent_messages)
        else:
            summary = self._summarizer.summarize(recent_messages)

        item = self.add_summary(
            summary,
            tags=["conversation", "auto_summary"],
            source="auto_summarize",
            importance=0.55,
        )
        return item.content if item is not None else summary

    def clear(self):
        """清空长期记忆"""
        self.items.clear()
        self._save()

    def _find_duplicate(self, item: MemoryItem) -> Optional[MemoryItem]:
        for existing in self.items:
            if not existing.active or existing.type != item.type:
                continue
            if existing.content_hash == item.content_hash:
                return existing
            existing_vec, _ = self._ensure_item_embedding(existing, preferred_kind=self._embedding_kind(item))
            if _cosine(existing_vec, item.embedding or {}) >= self.similarity_threshold:
                return existing
        return None

    def _embed_text(self, text: str) -> tuple[Any, str]:
        if self.embedder is not None and hasattr(self.embedder, "encode_query"):
            try:
                return _dense_vector(self.embedder.encode_query(text)), "rag"
            except Exception as exc:
                logger.warning(f"RAG embedding 生成失败，回退到本地记忆向量: {exc}")
        return _text_vector(text), "local"

    def _refresh_item_embedding(self, item: MemoryItem) -> None:
        vector, kind = self._embed_text(item.content)
        item.embedding = vector
        item.metadata["embedding_kind"] = kind
        if self.embedder is not None and getattr(self.embedder, "model_name", None):
            item.metadata["embedding_model"] = self.embedder.model_name

    def _ensure_item_embedding(self, item: MemoryItem, preferred_kind: Optional[str] = None) -> tuple[Any, bool]:
        current_kind = item.metadata.get("embedding_kind")
        expected_kind = preferred_kind or ("rag" if self.embedder is not None else "local")
        if item.embedding is None or current_kind != expected_kind:
            self._refresh_item_embedding(item)
            return item.embedding or {}, True
        return item.embedding or {}, False

    @staticmethod
    def _embedding_kind(item: MemoryItem) -> Optional[str]:
        return item.metadata.get("embedding_kind")

    @staticmethod
    def _merge_item(existing: MemoryItem, incoming: MemoryItem) -> None:
        existing.importance = max(existing.importance, incoming.importance)
        existing.tags = sorted(set(existing.tags) | set(incoming.tags))
        existing.updated_at = datetime.now().timestamp()
        existing.metadata.update(incoming.metadata)
        if incoming.source and not existing.source:
            existing.source = incoming.source

    def _trim(self) -> None:
        if len(self.items) <= self.max_items:
            return
        active = [item for item in self.items if item.active]
        inactive = [item for item in self.items if not item.active]
        active.sort(key=lambda item: (item.importance, item.updated_at), reverse=True)
        self.items = active[:self.max_items] + inactive[-50:]

    @staticmethod
    def _matches_filters(item: MemoryItem, filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        memory_type = filters.get("type")
        if memory_type and item.type != memory_type:
            return False
        tags = filters.get("tags")
        if tags:
            required = {tags} if isinstance(tags, str) else set(tags)
            if not required.intersection(item.tags):
                return False
        source = filters.get("source")
        if source and item.source != source:
            return False
        return True

    @staticmethod
    def _term_overlap(query_terms: set[str], item_terms: set[str]) -> float:
        if not query_terms or not item_terms:
            return 0.0
        return len(query_terms.intersection(item_terms)) / max(1, len(query_terms))

    @staticmethod
    def _recency_score(item: MemoryItem) -> float:
        age_seconds = max(0.0, datetime.now().timestamp() - item.updated_at)
        age_days = age_seconds / 86400
        return 1.0 / (1.0 + age_days)


def _content_hash(text: str) -> str:
    normalized = _normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]", str(text).lower())
    stopwords = {
        "the", "a", "an", "and", "or", "to", "of", "in", "is", "are",
        "了", "的", "是", "在", "和", "或", "为", "与", "将", "已",
    }
    return [token for token in tokens if token not in stopwords]


def _text_vector(text: str) -> Dict[str, float]:
    counts: Dict[str, float] = {}
    for token in _tokenize(text):
        counts[token] = counts.get(token, 0.0) + 1.0
    norm = math.sqrt(sum(value * value for value in counts.values()))
    if norm == 0:
        return {}
    return {token: value / norm for token, value in counts.items()}


def _cosine(left: Any, right: Any) -> float:
    if not left or not right:
        return 0.0
    if isinstance(left, list) and isinstance(right, list):
        return _dense_cosine(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return _sparse_cosine(left, right)
    return 0.0


def _sparse_cosine(left: Dict[str, float], right: Dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


def _dense_vector(values: Any) -> List[float]:
    if values is None:
        return []
    result = []
    for value in values:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            continue
    norm = math.sqrt(sum(value * value for value in result))
    if norm == 0:
        return result
    return [value / norm for value in result]


def _dense_cosine(left: List[float], right: List[float]) -> float:
    size = min(len(left), len(right))
    if size == 0:
        return 0.0
    return sum(left[i] * right[i] for i in range(size))


def _token_set(text: str) -> set[str]:
    return set(_tokenize(text))


def _infer_tags(content: str, memory_type: str) -> List[str]:
    tags = {memory_type}
    lower = content.lower()
    if any(word in lower for word in ("test", "pytest", "compileall", "测试", "验证", "通过")):
        tags.add("test")
    if any(word in lower for word in ("error", "traceback", "exception", "错误", "失败", "权限拒绝")):
        tags.add("error")
    if any(word in lower for word in ("file", "read_file", "write_file", "edit_file", ".py", ".md", "文件")):
        tags.add("file")
    if any(word in lower for word in ("todo", "planning", "计划", "任务")):
        tags.add("planning")
    return sorted(tags)


def _shorten(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def _clamp_float(value: float, low: float, high: float) -> float:
    try:
        return min(max(float(value), low), high)
    except (TypeError, ValueError):
        return low
