"""Context compression for long conversations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

from core.message import Message

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    summary: str
    messages: List[Message]
    compressed_count: int
    before_tokens: int
    after_tokens: int


class ContextCompressor:
    """Compacts old short-memory messages into a single system summary."""

    def __init__(
        self,
        threshold_tokens: int = 50000,
        keep_recent: int = 12,
        summary_max_chars: int = 4000,
    ):
        if keep_recent < 2:
            raise ValueError("keep_recent must be >= 2")
        self.threshold_tokens = threshold_tokens
        self.keep_recent = keep_recent
        self.summary_max_chars = summary_max_chars

    def should_compress(self, messages: List[Message]) -> bool:
        if len(messages) <= self.keep_recent:
            return False
        return self._total_tokens(messages) > self.threshold_tokens

    def compress(self, messages: List[Message], llm_client: Optional[Any] = None) -> CompressionResult:
        if len(messages) <= self.keep_recent:
            return CompressionResult(
                summary="",
                messages=list(messages),
                compressed_count=0,
                before_tokens=self._total_tokens(messages),
                after_tokens=self._total_tokens(messages),
            )

        before_tokens = self._total_tokens(messages)
        old_messages = messages[:-self.keep_recent]
        recent_messages = messages[-self.keep_recent:]
        summary = self._summarize(old_messages, llm_client)
        summary_message = self._build_summary_message(summary, len(old_messages), before_tokens)
        new_messages = [summary_message] + list(recent_messages)
        after_tokens = self._total_tokens(new_messages)

        if after_tokens >= before_tokens:
            summary = self._fit_summary(summary, recent_messages, before_tokens)
            summary_message = self._build_summary_message(summary, len(old_messages), before_tokens)
            new_messages = [summary_message] + list(recent_messages)
            after_tokens = self._total_tokens(new_messages)

        return CompressionResult(
            summary=summary,
            messages=new_messages,
            compressed_count=len(old_messages),
            before_tokens=before_tokens,
            after_tokens=after_tokens,
        )

    @staticmethod
    def _build_summary_message(summary: str, compressed_count: int, before_tokens: int) -> Message:
        return Message(
            role="system",
            content=(
                "【上下文压缩摘要】\n"
                "以下内容是较早对话和工具结果的压缩摘要，用于延续当前任务：\n"
                f"{summary}"
            ),
            metadata={
                "context_compression": True,
                "compressed_count": compressed_count,
                "before_tokens": before_tokens,
            },
        )

    def _fit_summary(self, summary: str, recent_messages: List[Message], before_tokens: int) -> str:
        """Shrink summary if compaction would otherwise increase context size."""
        recent_tokens = self._total_tokens(recent_messages)
        target_tokens = max(80, int((before_tokens - recent_tokens) * 0.6))
        fitted = summary.strip()

        while fitted and Message.estimate_tokens(fitted) > target_tokens and len(fitted) > 240:
            fitted = self._shorten(fitted, max(240, int(len(fitted) * 0.65)))

        if Message.estimate_tokens(fitted) > target_tokens:
            fitted = self._shorten(fitted, 240)

        return fitted or "旧上下文已压缩；无可提取的详细摘要。"

    def _summarize(self, messages: List[Message], llm_client: Optional[Any]) -> str:
        if llm_client is not None:
            try:
                summary = self._summarize_with_llm(messages, llm_client)
                if summary:
                    return self._limit_summary(summary)
            except Exception as exc:
                logger.warning(f"LLM 上下文压缩失败，回退到本地摘要: {exc}")

        return self._fallback_summary(messages)

    def _summarize_with_llm(self, messages: List[Message], llm_client: Any) -> Optional[str]:
        if not hasattr(llm_client, "generate"):
            return None

        transcript = self._format_messages(messages, max_chars=12000)
        prompt = (
            "请把以下较早对话压缩成给代码 Agent 继续工作的摘要。\n"
            "必须保留：用户目标、已读文件、已修改文件、工具执行结果、错误信息、测试结论、未完成事项。\n"
            "不要添加没有依据的新事实。摘要控制在 1200 字以内。\n\n"
            f"{transcript}"
        )
        response = llm_client.generate(messages=[Message(role="user", content=prompt)])
        if isinstance(response, str):
            return response.strip()
        if isinstance(response, dict):
            text = response.get("text")
            return text.strip() if isinstance(text, str) else None
        return None

    def _fallback_summary(self, messages: List[Message]) -> str:
        counts = {}
        lines = []
        tool_results = []

        for msg in messages:
            counts[msg.role] = counts.get(msg.role, 0) + 1
            if msg.role == "user":
                lines.append(f"用户请求: {self._shorten(msg.content, 220)}")
            elif msg.role == "assistant" and msg.metadata.get("function_calls"):
                names = [c.get("name", "") for c in msg.metadata.get("function_calls", [])]
                lines.append(f"助手调用工具: {', '.join(n for n in names if n)}")
            elif msg.role == "assistant" and msg.metadata.get("function_call"):
                call = msg.metadata["function_call"]
                lines.append(f"助手调用工具: {call.get('name', '')}")
            elif msg.role == "tool":
                tool_name = msg.name or "tool"
                tool_results.append(f"{tool_name}: {self._shorten(msg.content, 260)}")
            elif msg.role == "assistant":
                lines.append(f"助手回复: {self._shorten(msg.content, 220)}")

        header = "压缩了 " + ", ".join(f"{role}={count}" for role, count in sorted(counts.items()))
        if tool_results:
            lines.append("工具结果摘要:")
            lines.extend(f"- {item}" for item in tool_results[-8:])

        summary = header + "\n" + "\n".join(lines[-24:])
        return self._limit_summary(summary)

    def _limit_summary(self, summary: str) -> str:
        summary = summary.strip()
        if len(summary) <= self.summary_max_chars:
            return summary
        return summary[: self.summary_max_chars] + "..."
    
    @staticmethod
    def _format_messages(messages: List[Message], max_chars: int) -> str:
        parts = []
        total = 0
        for msg in messages:
            name = f" name={msg.name}" if msg.name else ""
            block = f"[{msg.role}{name}]\n{msg.content}\n"
            if total + len(block) > max_chars:
                remaining = max_chars - total
                if remaining > 100:
                    parts.append(block[:remaining] + "\n...(截断)")
                break
            parts.append(block)
            total += len(block)
        return "\n".join(parts)

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        compact = " ".join(str(text).split())
        if len(compact) <= limit:
            return compact
        return compact[:limit] + "..."

    @staticmethod
    def _total_tokens(messages: List[Message]) -> int:
        return sum(msg.token_count() for msg in messages)
