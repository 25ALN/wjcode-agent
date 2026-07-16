"""Runtime-owned project analysis workflow state."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

NOT_STARTED = "Not Started"
SEARCHING = "Searching"
COMPLETED = "Completed"
SKIPPED = "Skipped"


@dataclass
class ChecklistItem:
    name: str
    label: str
    required: bool = True
    status: str = NOT_STARTED
    evidence_count: int = 0
    attempts: int = 0


@dataclass
class EvidenceItem:
    target: str
    tool_name: str
    args: Dict[str, Any]
    path: str
    content: str
    symbols: List[str] = field(default_factory=list)

    def label(self) -> str:
        if self.path:
            return self.path
        return f"{self.tool_name}:{self.target}"


class Checklist:
    """Concrete project-analysis checklist, not a prompt-only instruction."""

    REQUIRED_TARGETS = {"Architecture", "Runtime", "Memory", "Planning", "Prompt"}

    def __init__(self, target_names: Iterable[str]):
        self.items: Dict[str, ChecklistItem] = {}
        for name in target_names:
            self.items[name] = ChecklistItem(
                name=name,
                label=name,
                required=name in self.REQUIRED_TARGETS,
            )

    def mark_searching(self, name: str) -> None:
        item = self.items.get(name)
        if item and item.status == NOT_STARTED:
            item.status = SEARCHING

    def add_evidence(self, name: str, complete: bool = True) -> None:
        item = self.items.get(name)
        if not item:
            return
        item.evidence_count += 1
        item.status = COMPLETED if complete else SEARCHING

    def add_attempt(self, name: str) -> None:
        item = self.items.get(name)
        if item:
            item.attempts += 1
            if item.status == NOT_STARTED:
                item.status = SEARCHING

    def skip_if_empty(self, name: str) -> None:
        item = self.items.get(name)
        if item and item.evidence_count <= 0:
            item.status = SKIPPED

    def completed_ratio(self) -> float:
        if not self.items:
            return 1.0
        finished = sum(1 for item in self.items.values() if item.status in {COMPLETED, SKIPPED})
        return finished / len(self.items)

    def missing_required(self) -> List[str]:
        return [
            item.name
            for item in self.items.values()
            if item.required and item.status != COMPLETED
        ]

    def unfinished(self) -> List[str]:
        return [
            item.name
            for item in self.items.values()
            if item.status not in {COMPLETED, SKIPPED}
        ]

    def format(self) -> str:
        lines = []
        for item in self.items.values():
            lines.append(
                f"- {item.name}: {item.status}; evidence={item.evidence_count}; attempts={item.attempts}"
            )
        return "\n".join(lines)


class EvidenceStore:
    """Stores evidence collected by read-only project analysis tools."""

    def __init__(self):
        self.items: List[EvidenceItem] = []

    def add(self, item: EvidenceItem) -> None:
        self.items.append(item)

    def count_for(self, target: str) -> int:
        return sum(1 for item in self.items if item.target == target)

    def digest(self, limit: int = 20000) -> str:
        parts = []
        for index, item in enumerate(self.items, 1):
            content = item.content.strip()
            if len(content) > 1800:
                content = content[:1800] + "\n...(evidence truncated)"
            symbols = f"; symbols={', '.join(item.symbols[:8])}" if item.symbols else ""
            parts.append(
                f"### Evidence {index}: {item.target} | {item.tool_name} | {item.label()}{symbols}\n{content}"
            )
        digest = "\n\n".join(parts)
        if len(digest) > limit:
            digest = digest[:limit] + "\n...(evidence digest truncated)"
        return digest

    def labels(self) -> List[str]:
        labels = []
        for item in self.items:
            label = item.label()
            if label and label not in labels:
                labels.append(label)
        return labels

    def labels_for(self, target: str, limit: int = 6) -> List[str]:
        labels = []
        for item in self.items:
            if item.target != target:
                continue
            label = item.label()
            if label and label not in labels:
                labels.append(label)
            if len(labels) >= limit:
                break
        return labels

    def content_for(self, target: str, limit: int = 2400) -> str:
        chunks = []
        total = 0
        for item in self.items:
            if item.target != target:
                continue
            content = item.content.strip()
            if not content:
                continue
            budget = max(0, limit - total)
            if budget <= 0:
                break
            chunks.append(content[:budget])
            total += min(len(content), budget)
        return "\n".join(chunks)


class CoverageTracker:
    """Tracks visited project surface during analysis."""

    def __init__(self):
        self.visited_modules: Set[str] = set()
        self.visited_paths: Set[str] = set()
        self.visited_files: Set[str] = set()
        self.visited_symbols: Set[str] = set()

    def update(self, target: str, tool_name: str, args: Dict[str, Any], content: str) -> List[str]:
        self.visited_modules.add(target)
        path = str(args.get("path") or args.get("file_path") or "").strip().replace("\\", "/")
        if path:
            self.visited_paths.add(path)
            if "." in os.path.basename(path):
                self.visited_files.add(path)

        for found in self.extract_paths(content):
            self.visited_paths.add(found)
            if "." in os.path.basename(found):
                self.visited_files.add(found)

        symbols = self.extract_symbols(content)
        self.visited_symbols.update(symbols)
        return symbols

    @staticmethod
    def extract_paths(text: str) -> List[str]:
        patterns = (
            r"文件:\s*([^\n]+)",
            r"目录:\s*([^\n]+)",
            r"(?:^|\n)\s*([A-Za-z0-9_./-]+\.(?:py|md|txt|js|css|json|html|toml|yaml|yml))",
            r"(?:^|\n)\s*((?:core|tools|server|web|rag|llm)/[A-Za-z0-9_./-]+)",
        )
        found: List[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, str(text or "")):
                value = match.group(1).strip().split("(", 1)[0].strip()
                if value and value not in found:
                    found.append(value)
        return found

    @staticmethod
    def extract_symbols(text: str) -> List[str]:
        symbols = []
        patterns = (
            r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, str(text or ""), flags=re.MULTILINE):
                symbols.append(match.group(1))
        return symbols[:40]


class SearchPlanner:
    """Deterministic search planner for project analysis sessions."""

    TARGET_ORDER = (
        "Architecture",
        "CLI",
        "Prompt",
        "Runtime",
        "Planning",
        "Memory",
        "Context",
        "Permission",
        "Tool",
        "Server",
        "Web",
        "RAG",
        "Tests",
        "Deployment",
    )

    def __init__(self, objective: str, target_path: str):
        self.objective = objective
        self.target_path = target_path or "."
        self.current_target: Optional[str] = None
        self.next_target: Optional[str] = None
        self.attempted: Set[str] = set()
        self.plans = self._build_plans()

    def target_names(self) -> Tuple[str, ...]:
        return self.TARGET_ORDER

    def next_call(self, checklist: Checklist, available_tools: Iterable[str]) -> Optional[Dict[str, Any]]:
        available = set(available_tools or [])
        for target in self.TARGET_ORDER:
            item = checklist.items.get(target)
            if not item or item.status in {COMPLETED, SKIPPED}:
                continue
            self.current_target = target
            checklist.mark_searching(target)
            for tool_name, args in self.plans.get(target, ()):
                if tool_name not in available:
                    continue
                key = self._tool_key(target, tool_name, args)
                if key in self.attempted:
                    continue
                self.attempted.add(key)
                checklist.add_attempt(target)
                self.next_target = self._find_next_target(checklist, after=target)
                return {"name": tool_name, "args": dict(args), "analysis_target": target}
            if item.evidence_count > 0:
                item.status = COMPLETED
            else:
                checklist.skip_if_empty(target)
        self.current_target = None
        self.next_target = None
        return None

    def has_more(self, checklist: Checklist, available_tools: Iterable[str]) -> bool:
        available = set(available_tools or [])
        for target in self.TARGET_ORDER:
            item = checklist.items.get(target)
            if not item or item.status in {COMPLETED, SKIPPED}:
                continue
            for tool_name, args in self.plans.get(target, ()):
                if tool_name in available and self._tool_key(target, tool_name, args) not in self.attempted:
                    return True
        return False

    def _find_next_target(self, checklist: Checklist, after: str) -> Optional[str]:
        seen = False
        for target in self.TARGET_ORDER:
            if target == after:
                seen = True
                continue
            if not seen:
                continue
            item = checklist.items.get(target)
            if item and item.status not in {COMPLETED, SKIPPED}:
                return target
        return None

    def _build_plans(self) -> Dict[str, Tuple[Tuple[str, Dict[str, Any]], ...]]:
        if self.target_path != ".":
            return self._generic_plans(self.target_path)
        return {
            "Architecture": (
                ("ls", {"path": ".", "recursive": False, "max_entries": 160}),
                ("ls", {"path": "core", "recursive": False, "max_entries": 140}),
            ),
            "CLI": (
                ("read_file", {"path": "main.py", "start_line": 1, "end_line": 260}),
            ),
            "Prompt": (
                ("read_file", {"path": "AGENT.md", "start_line": 1, "end_line": 220}),
                ("read_file", {"path": "promote.txt", "start_line": 1, "end_line": 260}),
            ),
            "Runtime": (
                ("ls", {"path": "core/runtime", "recursive": False, "max_entries": 140}),
                ("read_file", {"path": "core/runtime/agent_runtime.py", "start_line": 1, "end_line": 260}),
                ("read_file", {"path": "core/runtime/agent_runtime.py", "start_line": 480, "end_line": 900}),
            ),
            "Planning": (
                ("ls", {"path": "core/planning", "recursive": False, "max_entries": 100}),
                ("read_file", {"path": "core/planning/manager.py", "start_line": 1, "end_line": 300}),
                ("ls", {"path": "core/todo", "recursive": False, "max_entries": 100}),
            ),
            "Memory": (
                ("ls", {"path": "core/memory", "recursive": False, "max_entries": 100}),
                ("read_file", {"path": "core/memory/store.py", "start_line": 1, "end_line": 320}),
            ),
            "Context": (
                ("ls", {"path": "core/context", "recursive": False, "max_entries": 120}),
                ("read_file", {"path": "core/context/builder.py", "start_line": 1, "end_line": 220}),
                ("read_file", {"path": "core/context/scratchpad.py", "start_line": 1, "end_line": 240}),
                ("read_file", {"path": "core/context/compression.py", "start_line": 1, "end_line": 260}),
            ),
            "Permission": (
                ("ls", {"path": "core/permission", "recursive": False, "max_entries": 100}),
                ("read_file", {"path": "core/permission/manager.py", "start_line": 1, "end_line": 260}),
                ("read_file", {"path": "core/permission/web.py", "start_line": 1, "end_line": 240}),
            ),
            "Tool": (
                ("ls", {"path": "tools", "recursive": False, "max_entries": 160}),
                ("read_file", {"path": "tools/registry.py", "start_line": 1, "end_line": 220}),
                ("read_file", {"path": "tools/file_tool.py", "start_line": 1, "end_line": 260}),
                ("read_file", {"path": "tools/project_tools.py", "start_line": 1, "end_line": 260}),
            ),
            "Server": (
                ("ls", {"path": "server", "recursive": False, "max_entries": 120}),
                ("read_file", {"path": "server/service.py", "start_line": 1, "end_line": 260}),
                ("read_file", {"path": "server/app.py", "start_line": 1, "end_line": 240}),
            ),
            "Web": (
                ("ls", {"path": "web", "recursive": False, "max_entries": 100}),
                ("read_file", {"path": "web/app.js", "start_line": 1, "end_line": 260}),
                ("read_file", {"path": "web/styles.css", "start_line": 1, "end_line": 220}),
            ),
            "RAG": (
                ("ls", {"path": "rag", "recursive": False, "max_entries": 100}),
                ("ls", {"path": "llm", "recursive": False, "max_entries": 100}),
                ("read_file", {"path": "rag/retriever.py", "start_line": 1, "end_line": 260}),
                ("read_file", {"path": "llm/deepseek_client.py", "start_line": 1, "end_line": 260}),
            ),
            "Tests": (
                ("read_file", {"path": "test_runtime.py", "start_line": 1, "end_line": 220}),
                ("read_file", {"path": "test_stage3_server.py", "start_line": 1, "end_line": 220}),
                ("read_file", {"path": "test_stage2.py", "start_line": 1, "end_line": 180}),
            ),
            "Deployment": (
                ("read_file", {"path": "README.md", "start_line": 1, "end_line": 180}),
                ("read_file", {"path": "requirements.txt", "start_line": 1, "end_line": 160}),
                ("read_file", {"path": "pyproject.toml", "start_line": 1, "end_line": 180}),
            ),
        }

    @staticmethod
    def _generic_plans(base: str) -> Dict[str, Tuple[Tuple[str, Dict[str, Any]], ...]]:
        base = base.strip().rstrip("/") or "."

        def join(path: str) -> str:
            return f"{base}/{path}" if base != "." else path

        return {
            "Architecture": (
                ("ls", {"path": base, "recursive": False, "max_entries": 180}),
                ("ls", {"path": base, "recursive": True, "max_entries": 260}),
            ),
            "CLI": (("read_file", {"path": join("main.py"), "start_line": 1, "end_line": 260}),),
            "Prompt": (
                ("read_file", {"path": join("AGENT.md"), "start_line": 1, "end_line": 200}),
                ("read_file", {"path": join("README.md"), "start_line": 1, "end_line": 220}),
            ),
            "Runtime": (("grep", {"pattern": r"(class|def|function|func) ", "path": base, "file_pattern": "*", "max_results": 120}),),
            "Planning": (("grep", {"pattern": r"(plan|todo|task)", "path": base, "file_pattern": "*", "max_results": 80}),),
            "Memory": (("grep", {"pattern": r"(memory|cache|history|session)", "path": base, "file_pattern": "*", "max_results": 80}),),
            "Context": (("grep", {"pattern": r"(context|prompt|scratchpad)", "path": base, "file_pattern": "*", "max_results": 80}),),
            "Permission": (("grep", {"pattern": r"(permission|auth|approve|risk)", "path": base, "file_pattern": "*", "max_results": 80}),),
            "Tool": (("grep", {"pattern": r"(tool|command|execute)", "path": base, "file_pattern": "*", "max_results": 80}),),
            "Server": (("grep", {"pattern": r"(server|route|api|fastapi|express)", "path": base, "file_pattern": "*", "max_results": 80}),),
            "Web": (("grep", {"pattern": r"(web|frontend|html|css|js)", "path": base, "file_pattern": "*", "max_results": 80}),),
            "RAG": (("grep", {"pattern": r"(rag|embedding|retriever|vector)", "path": base, "file_pattern": "*", "max_results": 80}),),
            "Tests": (
                ("ls", {"path": join("tests"), "recursive": False, "max_entries": 120}),
                ("grep", {"pattern": r"(pytest|unittest|jest|vitest|go test|cargo test)", "path": base, "file_pattern": "*", "max_results": 100}),
            ),
            "Deployment": (
                ("read_file", {"path": join("requirements.txt"), "start_line": 1, "end_line": 160}),
                ("read_file", {"path": join("package.json"), "start_line": 1, "end_line": 180}),
                ("read_file", {"path": join("pyproject.toml"), "start_line": 1, "end_line": 180}),
            ),
        }

    @staticmethod
    def _tool_key(target: str, name: str, args: Dict[str, Any]) -> str:
        parts = [target, name]
        for key in sorted(args):
            parts.append(f"{key}={args[key]}")
        return "|".join(parts)


class AnalysisSession:
    """Full runtime workflow for codebase/repository analysis."""

    def __init__(self, goal: str):
        self.goal = str(goal or "").strip()
        self.target_path = self._detect_target_path(self.goal)
        self.search_planner = SearchPlanner(self.goal, self.target_path)
        self.checklist = Checklist(self.search_planner.target_names())
        self.evidence_store = EvidenceStore()
        self.coverage_tracker = CoverageTracker()
        self.analysis_notes: List[str] = []
        self.tool_results: List[str] = []

    @property
    def current_target(self) -> Optional[str]:
        return self.search_planner.current_target

    @property
    def next_target(self) -> Optional[str]:
        return self.search_planner.next_target

    @property
    def coverage_ratio(self) -> float:
        return self.checklist.completed_ratio()

    def next_tool_call(self, available_tools: Iterable[str]) -> Optional[Dict[str, Any]]:
        return self.search_planner.next_call(self.checklist, available_tools)

    def has_more_search(self, available_tools: Iterable[str]) -> bool:
        return self.search_planner.has_more(self.checklist, available_tools)

    def observe_tool(self, call: Dict[str, Any], content: str) -> None:
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        target = str(call.get("analysis_target") or self.current_target or "Architecture")
        tool_name = str(call.get("name") or "")
        text = str(content or "")
        self.tool_results.append(text[:5000])

        symbols = self.coverage_tracker.update(target, tool_name, args, text)
        if self._is_low_information_result(text):
            self.checklist.skip_if_empty(target)
            return

        path = str(args.get("path") or args.get("file_path") or "").strip()
        item = EvidenceItem(
            target=target,
            tool_name=tool_name,
            args=dict(args),
            path=path,
            content=text[:5000],
            symbols=symbols,
        )
        self.evidence_store.add(item)
        self.checklist.add_evidence(target, complete=self._target_has_enough_evidence(target, call))

    @staticmethod
    def _target_has_enough_evidence(target: str, call: Dict[str, Any]) -> bool:
        tool_name = str(call.get("name") or "")
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        path = str(args.get("path") or args.get("file_path") or "").replace("\\", "/").lower()
        if target == "Architecture":
            return tool_name == "ls"
        if target == "Runtime":
            return tool_name in {"read_file", "grep"} and ("agent_runtime.py" in path or tool_name == "grep")
        if target == "Planning":
            return tool_name in {"read_file", "grep"} and ("planning" in path or "todo" in path or tool_name == "grep")
        if target == "Memory":
            return tool_name in {"read_file", "grep"} and ("memory" in path or tool_name == "grep")
        if target == "Context":
            return tool_name in {"read_file", "grep"} and ("context" in path or tool_name == "grep")
        if target == "Prompt":
            return tool_name == "read_file"
        if target in {"Permission", "Tool", "Server", "Web", "RAG", "Tests", "Deployment", "CLI"}:
            return tool_name in {"read_file", "grep"}
        return tool_name in {"read_file", "grep"}

    def ready_for_summary(self) -> bool:
        return not self.checklist.missing_required() and not self.has_more_search(("ls", "grep", "read_file"))

    def missing_for_final(self) -> List[str]:
        missing = self.checklist.missing_required()
        if missing:
            return missing
        return self.checklist.unfinished()

    def final_has_evidence(self, text: str) -> bool:
        raw = str(text or "")
        lowered = raw.lower()
        if not any(marker in lowered for marker in ("evidence", "依据", "证据", "参考", "文件")):
            return False
        labels = self.evidence_store.labels()
        if any(label and label.lower() in lowered for label in labels):
            return True
        return bool(re.search(
            r"(\b[A-Za-z0-9_.-]+/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+|"
            r"\b[A-Za-z0-9_.-]+\.(?:py|md|txt|js|css|json|html|toml|yaml|yml))",
            raw,
        ))

    def checkpoint_context(self, base_context: List[Any], user_input: str) -> List[Any]:
        from core.message import Message

        return self._state_context(base_context, user_input, phase="analysis") + [
            Message(role="user", content="请只根据当前 Analysis State 给出简短阶段性判断；不要输出最终项目总结。")
        ]

    def final_context(self, base_context: List[Any], user_input: str) -> List[Any]:
        from core.message import Message

        return self._state_context(base_context, user_input, phase="final") + [
            Message(role="user", content="请基于 Analysis State 和 Evidence Store 输出最终项目分析。")
        ]

    def state_message(self, phase: str = "analysis", evidence_limit: int = 9000) -> Any:
        from core.message import Message

        missing = ", ".join(self.missing_for_final()) or "none"
        content = (
            f"【Project Analysis State】\n"
            f"Phase: {phase}\n"
            f"Goal: {self.goal}\n"
            f"Target Path: {self.target_path}\n"
            f"Current Target: {self.current_target or 'none'}\n"
            f"Next Target: {self.next_target or 'none'}\n"
            f"Coverage: {self.coverage_ratio:.0%}\n"
            f"Missing For Final: {missing}\n\n"
            f"Checklist:\n{self.checklist.format()}\n\n"
            f"Visited Modules: {', '.join(sorted(self.coverage_tracker.visited_modules)) or 'none'}\n"
            f"Visited Files: {', '.join(sorted(self.coverage_tracker.visited_files)[:20]) or 'none'}\n"
            f"Visited Symbols: {', '.join(sorted(self.coverage_tracker.visited_symbols)[:30]) or 'none'}\n\n"
            f"Evidence Store:\n{self.evidence_store.digest(limit=evidence_limit)}"
        )
        return Message(role="system", content=content)

    def reflection_message(self) -> Any:
        from core.message import Message

        missing = ", ".join(self.missing_for_final()) or "none"
        return Message(
            role="system",
            content=(
                "【Project Analysis Runtime Reflection】\n"
                f"Runtime refused final because these checklist items are missing: {missing}.\n"
                f"Current Target: {self.current_target or 'none'}; Next Target: {self.next_target or 'none'}."
            ),
        )

    def fallback_answer(self) -> str:
        labels = self.evidence_store.labels()
        evidence_line = "、".join(labels[:14]) if labels else "已收集的项目结构和关键文件信息"
        capability_text = self._capability_summary()
        grouped = self._grouped_evidence_lines()

        return (
            "已完成项目分析流程。DeepSeek 最终总结不可用时，以下是 Runtime 基于 EvidenceStore 生成的详细分析。\n\n"
            "**项目定位与架构**\n"
            "- 这个项目不是普通聊天壳，而是一个 code agent runtime：核心围绕 `AgentRuntime`、工具注册/执行、上下文构建、记忆、规划、权限、Web/SSE 会话和前端工作台组织。\n"
            "- 主链路可以概括为：用户请求进入 Runtime，按意图选择普通回答、实时搜索、工具执行或项目分析；复杂执行任务接入 Planning/Todo；项目分析任务由 Runtime 的 AnalysisSession/SearchPlanner 主导搜索。\n"
            f"- 当前 Evidence 覆盖了：{capability_text}。\n\n"
            "**亮点**\n"
            "- Runtime 边界比较清晰：工具协议、权限恢复、普通问答、项目分析、事件流和最终回答分别有独立路径，降低了单一 prompt 承担全部控制逻辑的风险。\n"
            "- 工具系统已经具备 code agent 需要的基础能力：`ls`/`grep`/`read_file` 用于理解项目，`edit_file`/`write_file` 用于修改，`execute_code` 用于验证，`web_search` 用于实时信息。\n"
            "- Context 体系不是单一历史拼接，已经包含 ShortMemory、LongMemory、Compression、Scratchpad、Project Context、Planning/Todo 等层次。\n"
            "- 权限系统和 Web 权限恢复是亮点：风险工具不会静默执行，并且恢复时会补齐同批 tool calls，能避免 OpenAI/DeepSeek function calling 协议错误。\n"
            "- Web 层使用结构化 SSE 事件，把 user/final/tool_call/tool_result/permission/todo/planning 分开，方向上更接近真实 agent UI。\n"
            "- 新的项目分析流程由 Runtime 主导 Evidence Collection 和 Final Gate，比让模型读几份文件后自由 Final 更稳。\n\n"
            "**难点与风险**\n"
            "- 最大难点是 Runtime 工作流控制：普通聊天、项目分析、复杂执行、权限恢复、实时搜索都走不同策略，任何分支误判都会表现为误调工具、卡住或回答变浅。\n"
            "- Function Calling 协议要求严格：assistant 只要带 tool_calls，下一次模型请求前必须补齐每个 tool_call_id 的 tool message；权限挂起/恢复尤其容易破坏这个顺序。\n"
            "- 上下文管理风险高：Tool Results、Scratchpad、Planning、Memory、Project Context 都可能膨胀；如果不压缩或不分层注入，很容易导致 API timeout 或模型输出伪工具文本。\n"
            "- 长期记忆和会话历史容易混淆：history 负责恢复可见会话，LongMemory 负责结构化检索；两者边界不清会让追问质量下降。\n"
            "- Web 端要保持事件和最终回答分离，否则用户会看到 read_file/tool_calls/thinking 混在答案里。\n"
            "- 项目分析的质量取决于 Evidence 覆盖面和本地兜底总结质量；一旦最终 LLM 超时，本地总结必须足够详细，不能只给两三条概括。\n\n"
            "**改进建议**\n"
            "- 项目分析阶段尽量减少中间 LLM 调用，只在 final gate 使用紧凑 EvidenceStore 调模型；失败时使用结构化本地总结，避免把 API timeout 暴露给用户。\n"
            "- 给 AnalysisSession 的 Evidence 增加更细的摘要字段，例如 per-target summary、risk tags、important symbols，后续最终总结可以更具体。\n"
            "- 对 follow-up 问题增加基于上一轮项目分析的本地展开能力，避免用户要求“详细说说”时再次因为模型超时而无回答。\n"
            "- 为 Web UI 增加项目分析 coverage 面板，让用户能看到 Architecture/Runtime/Memory/Planning 等 checklist 的完成状态。\n\n"
            f"**Evidence**：{evidence_line}。\n"
            f"\n**Evidence 分组**\n{grouped}"
        )

    def _grouped_evidence_lines(self) -> str:
        lines = []
        for target in self.search_planner.TARGET_ORDER:
            labels = self.evidence_store.labels_for(target, limit=5)
            if labels:
                lines.append(f"- {target}: {', '.join(labels)}")
        return "\n".join(lines) if lines else "- 暂无可分组 Evidence"

    def _state_context(self, base_context: List[Any], user_input: str, phase: str) -> List[Any]:
        from core.message import Message

        # Project analysis finalization must be compact. Re-sending full chat,
        # tool protocol, AGENT.md, and large evidence bodies is a common cause of
        # API read timeouts. Keep only the minimal instruction surface plus the
        # runtime-owned analysis state.
        compact: List[Any] = [Message(
            role="system",
            content=(
                "你是一个 code agent 的项目分析器。只基于 Runtime 提供的 Analysis State "
                "和 Evidence Store 输出结论，不要请求工具。"
            ),
        )]
        evidence_limit = 7000 if phase == "final" else 3500
        compact.append(self.state_message(phase=phase, evidence_limit=evidence_limit))
        if phase == "final":
            compact.append(Message(
                role="system",
                content=(
                    "【Project Analysis Final Gate】\n"
                    "Runtime has completed the analysis workflow. The final answer must be detailed and include: "
                    "architecture, highlights, difficulties/risks, improvement suggestions, and Evidence with concrete files/modules."
                ),
            ))
            compact.append(Message(role="user", content=f"用户目标：{user_input}\n请输出详细项目分析。"))
        else:
            compact.append(Message(role="user", content=f"用户目标：{user_input}\n请给出简短阶段性分析记录。"))
        return compact

    def _capability_summary(self) -> str:
        joined = "\n".join(item.content for item in self.evidence_store.items).lower()
        found = []
        checks = (
            ("Function Calling / Tool Loop", ("function_call", "tool_calls", "react")),
            ("Planning / Todo", ("planning", "todo", "planstate")),
            ("Memory / Context / Scratchpad", ("memory", "context", "scratchpad", "compression")),
            ("Permission", ("permission", "risk_level", "approve")),
            ("Server / Web", ("fastapi", "sse", "web", "app.js")),
            ("RAG / LLM", ("rag", "embedding", "deepseek", "retriever")),
        )
        for label, needles in checks:
            if any(needle in joined for needle in needles):
                found.append(label)
        return "、".join(found[:6]) if found else "Architecture、Runtime、Tool、Evidence Store"

    @staticmethod
    def _is_low_information_result(text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return True
        markers = (
            "[错误]",
            "文件不存在",
            "路径不存在",
            "路径不是文件",
            "路径不是目录",
            "未找到匹配结果",
            "permission denied",
            "not found",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _detect_target_path(objective: str) -> str:
        text = str(objective or "").strip()
        if not text:
            return "."
        skip = {"这个", "该", "当前", "整个", "本", "项目", "仓库", "目录", "代码库", "工程"}
        patterns = (
            r"([A-Za-z0-9_./-]+)\s*(?:这个|该)?(?:目录|文件夹|项目|仓库|代码库)",
            r"(?:目录|文件夹|项目|仓库|代码库)\s*[:：]?\s*([A-Za-z0-9_./-]+)",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                candidate = match.group(1).strip().strip(".,，。:：;；")
                if candidate and candidate not in skip:
                    return candidate
        return "."


class ProjectExplorer(AnalysisSession):
    """Backward-compatible name for project analysis sessions."""


__all__ = [
    "AnalysisSession",
    "Checklist",
    "ChecklistItem",
    "CoverageTracker",
    "EvidenceItem",
    "EvidenceStore",
    "ProjectExplorer",
    "SearchPlanner",
]
