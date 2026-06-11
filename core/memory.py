from typing import List, Optional, Dict, Any
from core.message import Message
import json
import os


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

        # 按消息条数截断
        while len(self.messages) > self.max_length:
            self.messages.pop(0)

        # 按 token 数截断（从最早开始丢弃）
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


class LongMemory:
    """长期记忆 - 管理持久化关键信息"""


    def __init__(self, storage_path: str = "memory_long.json"):
        """
        Args:
            storage_path: 长期记忆的 JSON 文件路径
        """
        self.storage_path = storage_path
        self.facts: List[str] = []
        self.summaries: List[str] = []
        self._load()

    def _load(self):
        """从文件加载长期记忆"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.facts = data.get("facts", [])
                    self.summaries = data.get("summaries", [])
            except (json.JSONDecodeError, IOError):
                self.facts = []
                self.summaries = []

    def _save(self):
        """保存长期记忆到文件"""
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump({
                "facts": self.facts,
                "summaries": self.summaries,
            }, f, ensure_ascii=False, indent=2)

    def add_fact(self, fact: str):
        """添加一个关键事实"""
        if fact not in self.facts:
            self.facts.append(fact)
            # 限制 facts 数量
            if len(self.facts) > 100:
                self.facts = self.facts[-100:]
            self._save()

    def add_summary(self, summary: str):
        """添加一次对话总结"""
        self.summaries.append(summary)
        if len(self.summaries) > 50:
            self.summaries = self.summaries[-50:]
        self._save()

    def get_context(self, max_facts: int = 10, max_summaries: int = 3) -> str:
        """获取长期记忆上下文文本"""
        parts = []
        if self.facts:
            facts_text = "\n".join(f"- {f}" for f in self.facts[-max_facts:])
            parts.append(f"【长期记忆 - 关键事实】\n{facts_text}")
        if self.summaries:
            summaries_text = "\n".join(f"- {s}" for s in self.summaries[-max_summaries:])
            parts.append(f"【历史对话摘要】\n{summaries_text}")
        return "\n\n".join(parts)

    def summarize_and_store(self, recent_messages: List[Message],
                            llm_summarize_fn=None) -> Optional[str]:
        if not recent_messages:
            return None

        if llm_summarize_fn:
            summary = llm_summarize_fn(recent_messages)
        else:
            # 简单总结：提取关键信息
            user_msgs = [m.content for m in recent_messages if m.role == "user"]
            assistant_msgs = [m.content for m in recent_messages if m.role == "assistant"]
            summary_parts = []
            if user_msgs:
                summary_parts.append(f"用户问了 {len(user_msgs)} 个问题")
            if assistant_msgs:
                summary_parts.append(f"助手回复了 {len(assistant_msgs)} 条消息")
            summary = "；".join(summary_parts) if summary_parts else "简短对话"

        self.add_summary(summary)
        return summary

    def clear(self):
        """清空长期记忆"""
        self.facts.clear()
        self.summaries.clear()
        self._save()