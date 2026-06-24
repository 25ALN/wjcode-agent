"""Scratchpad for compact task-local execution state.

The scratchpad stores explicit working-state facts for the current task. It is
not a hidden chain-of-thought store; entries are short observations that can be
shown to users or injected into context safely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


SCRATCHPAD_SECTIONS = (
    "objective",
    "facts",
    "files",
    "attempts",
    "blockers",
    "next_steps",
)


@dataclass
class Scratchpad:
    max_items_per_section: int = 8
    max_item_chars: int = 220
    objective: Optional[str] = None
    facts: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    attempts: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    updated_at: Optional[float] = None

    def set_objective(self, objective: str) -> None:
        text = self._normalize(objective, self.max_item_chars)
        if text and text != self.objective:
            self.objective = text
            self._touch()

    def add_fact(self, text: str) -> bool:
        return self._append_unique(self.facts, text)

    def add_file(self, path: str, note: Optional[str] = None) -> bool:
        path_text = self._normalize(path, 160)
        if not path_text:
            return False
        entry = path_text
        note_text = self._normalize(note or "", 80)
        if note_text:
            entry = f"{path_text} - {note_text}"
        return self._append_unique(self.files, entry)

    def add_attempt(self, text: str) -> bool:
        return self._append_unique(self.attempts, text)

    def add_blocker(self, text: str) -> bool:
        return self._append_unique(self.blockers, text)

    def add_next_step(self, text: str) -> bool:
        return self._append_unique(self.next_steps, text)

    def observe_tool_result(self, tool_name: str, args: dict, observation: str) -> None:
        """Record compact, explicit state from a tool observation."""
        args = args if isinstance(args, dict) else {}
        observation_text = self._normalize(observation, self.max_item_chars)
        path = args.get("path") or args.get("directory") or args.get("root")

        if tool_name in {"read_file", "write_file", "edit_file"} and path:
            self.add_file(str(path), self._tool_note(tool_name, observation_text))
        elif tool_name in {"ls", "grep"}:
            target = path or args.get("pattern") or args.get("query")
            if target:
                self.add_fact(f"{tool_name}: inspected {target}")

        if observation_text:
            self.add_attempt(f"{tool_name}: {observation_text}")

        if self._looks_like_error(observation_text):
            self.add_blocker(f"{tool_name}: {observation_text}")
            self.add_next_step("根据最新错误或权限结果调整方案并重新验证")
        elif tool_name in {"write_file", "edit_file", "execute_code"}:
            self.add_next_step("运行测试或静态检查验证最新修改")

    def merge_next_steps(self, steps: List[str]) -> None:
        for step in steps:
            self.add_next_step(step)

    def clear(self) -> None:
        self.objective = None
        self.facts.clear()
        self.files.clear()
        self.attempts.clear()
        self.blockers.clear()
        self.next_steps.clear()
        self.updated_at = None

    def is_empty(self) -> bool:
        return not any([
            self.objective,
            self.facts,
            self.files,
            self.attempts,
            self.blockers,
            self.next_steps,
        ])

    def format_for_prompt(self) -> Optional[str]:
        if self.is_empty():
            return None

        lines = []
        if self.objective:
            lines.append(f"目标: {self.objective}")
        self._append_section(lines, "已确认事实", self.facts)
        self._append_section(lines, "相关文件", self.files)
        self._append_section(lines, "已尝试操作", self.attempts)
        self._append_section(lines, "阻塞/风险", self.blockers)
        self._append_section(lines, "下一步", self.next_steps)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        return {
            "objective": self.objective,
            "facts": list(self.facts),
            "files": list(self.files),
            "attempts": list(self.attempts),
            "blockers": list(self.blockers),
            "next_steps": list(self.next_steps),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Scratchpad":
        scratchpad = cls()
        scratchpad.objective = data.get("objective") if isinstance(data.get("objective"), str) else None
        for attr in ("facts", "files", "attempts", "blockers", "next_steps"):
            values = data.get(attr, [])
            if isinstance(values, list):
                setattr(scratchpad, attr, [str(item) for item in values if str(item).strip()])
        updated_at = data.get("updated_at")
        scratchpad.updated_at = updated_at if isinstance(updated_at, (int, float)) else None
        return scratchpad

    def _append_unique(self, target: List[str], text: str) -> bool:
        entry = self._normalize(text, self.max_item_chars)
        if not entry:
            return False
        if entry in target:
            return False
        target.append(entry)
        if len(target) > self.max_items_per_section:
            del target[:-self.max_items_per_section]
        self._touch()
        return True

    def _touch(self) -> None:
        self.updated_at = datetime.now().timestamp()

    @staticmethod
    def _append_section(lines: List[str], title: str, values: List[str]) -> None:
        if not values:
            return
        lines.append(f"{title}:")
        lines.extend(f"- {value}" for value in values)

    @staticmethod
    def _normalize(text: str, limit: int) -> str:
        compact = " ".join(str(text or "").split())
        if len(compact) <= limit:
            return compact
        return compact[:limit] + "..."

    @staticmethod
    def _looks_like_error(text: str) -> bool:
        keywords = (
            "[错误]",
            "错误",
            "失败",
            "Traceback",
            "Exception",
            "Error",
            "failed",
            "FAIL",
            "权限拒绝",
        )
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _tool_note(tool_name: str, observation: str) -> str:
        if "[预览]" in observation:
            return f"{tool_name} dry-run preview"
        if "[成功]" in observation or "成功" in observation:
            return f"{tool_name} succeeded"
        if Scratchpad._looks_like_error(observation):
            return f"{tool_name} error"
        return tool_name
