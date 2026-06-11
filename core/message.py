from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from datetime import datetime
import json


@dataclass
class Message:
    role: str  # user | assistant | system | tool
    content: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None  # 用于 tool 调用时的函数名

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化的字典"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从字典反序列化"""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", datetime.now().timestamp()),
            metadata=data.get("metadata", {}),
            name=data.get("name"),
        )

    def summary(self, max_length: int = 80) -> str:
        """生成简短摘要（用于日志/调试）"""
        content_preview = self.content[:max_length]
        if len(self.content) > max_length:
            content_preview += "..."
        return f"[{self.role}] {content_preview}"

    @staticmethod
    def estimate_tokens(text: str) -> int:
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.25) + 4

    def token_count(self) -> int:
        """返回本条消息的估算 token 数"""
        return self.estimate_tokens(self.content)