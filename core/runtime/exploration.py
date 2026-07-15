"""Project exploration state for repository/codebase analysis turns."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class ExplorationStep:
    key: str
    label: str
    module_hints: Tuple[str, ...]
    tool_plan: Tuple[Tuple[str, Dict[str, Any]], ...]
    done: bool = False
    evidence: List[str] = field(default_factory=list)


@dataclass
class InformationGain:
    score: int
    new_modules: List[str] = field(default_factory=list)
    new_paths: List[str] = field(default_factory=list)
    new_symbols: List[str] = field(default_factory=list)


class ProjectExplorer:
    """Runtime-owned exploration state for repository analysis turns.

    The model can suggest read-only tool calls, but project/repository analysis is
    not allowed to end just because the model emitted a quick answer. The runtime
    tracks checklist coverage, information gain, and evidence quality, then asks
    the model to continue exploring until the collected observations are enough.
    """

    COVERAGE_THRESHOLD = 0.90
    MIN_EVIDENCE_ITEMS = 5
    LOW_GAIN_LIMIT = 2

    def __init__(self, objective: str):
        self.objective = str(objective or "").strip()
        self.target_path = self._detect_target_path(self.objective)
        self.steps: List[ExplorationStep] = self._default_steps(self.target_path)
        self.visited_modules: Set[str] = set()
        self.visited_paths: Set[str] = set()
        self.visited_files: Set[str] = set()
        self.visited_symbols: Set[str] = set()
        self.attempted_tools: Set[str] = set()
        self.observations: List[str] = []
        self.low_gain_streak = 0
        self.blocked_final_attempts = 0

    @classmethod
    def _default_steps(cls, target_path: str) -> List[ExplorationStep]:
        if target_path and target_path != ".":
            return cls._generic_project_steps(target_path)

        return [
            ExplorationStep(
                key="root",
                label="查看项目目录和顶层文件",
                module_hints=("agent.md", "promote.txt", "main.py", "server", "core", "tools", "web", "rag"),
                tool_plan=(("ls", {"path": ".", "recursive": False, "max_entries": 120}),),
            ),
            ExplorationStep(
                key="entry",
                label="查看核心入口和项目规则",
                module_hints=("main.py", "agent.md", "promote.txt", "readme"),
                tool_plan=(
                    ("read_file", {"path": "main.py", "start_line": 1, "end_line": 220}),
                    ("read_file", {"path": "AGENT.md", "start_line": 1, "end_line": 180}),
                    ("read_file", {"path": "promote.txt", "start_line": 1, "end_line": 220}),
                ),
            ),
            ExplorationStep(
                key="runtime",
                label="查看 Agent Runtime / Tool Loop",
                module_hints=("core/runtime", "agent_runtime.py", "run_events", "function_call"),
                tool_plan=(
                    ("ls", {"path": "core/runtime", "recursive": False, "max_entries": 120}),
                    ("read_file", {"path": "core/runtime/agent_runtime.py", "start_line": 1, "end_line": 260}),
                    ("read_file", {"path": "core/runtime/agent_runtime.py", "start_line": 480, "end_line": 760}),
                ),
            ),
            ExplorationStep(
                key="tools",
                label="查看工具系统",
                module_hints=("tools", "toolregistry", "read_file", "execute_code", "grep"),
                tool_plan=(
                    ("ls", {"path": "tools", "recursive": False, "max_entries": 120}),
                    ("read_file", {"path": "tools/registry.py", "start_line": 1, "end_line": 180}),
                    ("read_file", {"path": "tools/file_tool.py", "start_line": 1, "end_line": 210}),
                ),
            ),
            ExplorationStep(
                key="memory_context",
                label="查看 Memory / Context / Scratchpad",
                module_hints=("core/memory", "core/context", "scratchpad", "compression", "longmemory"),
                tool_plan=(
                    ("ls", {"path": "core/memory", "recursive": False, "max_entries": 80}),
                    ("ls", {"path": "core/context", "recursive": False, "max_entries": 100}),
                    ("read_file", {"path": "core/memory/store.py", "start_line": 1, "end_line": 260}),
                    ("read_file", {"path": "core/context/builder.py", "start_line": 1, "end_line": 220}),
                ),
            ),
            ExplorationStep(
                key="planning_todo",
                label="查看 Planning / Todo",
                module_hints=("core/planning", "core/todo", "planningmanager", "todolist"),
                tool_plan=(
                    ("ls", {"path": "core/planning", "recursive": False, "max_entries": 80}),
                    ("read_file", {"path": "core/planning/manager.py", "start_line": 1, "end_line": 260}),
                    ("ls", {"path": "core/todo", "recursive": False, "max_entries": 80}),
                ),
            ),
            ExplorationStep(
                key="permission",
                label="查看权限系统",
                module_hints=("core/permission", "web_permission", "permissionmanager", "permission"),
                tool_plan=(
                    ("ls", {"path": "core/permission", "recursive": False, "max_entries": 80}),
                    ("read_file", {"path": "core/permission/manager.py", "start_line": 1, "end_line": 240}),
                    ("read_file", {"path": "core/permission/web.py", "start_line": 1, "end_line": 220}),
                ),
            ),
            ExplorationStep(
                key="server_web",
                label="查看 Server / Web UI",
                module_hints=("server", "web", "sse", "fastapi", "app.js"),
                tool_plan=(
                    ("ls", {"path": "server", "recursive": False, "max_entries": 100}),
                    ("read_file", {"path": "server/service.py", "start_line": 1, "end_line": 220}),
                    ("ls", {"path": "web", "recursive": False, "max_entries": 80}),
                    ("read_file", {"path": "web/app.js", "start_line": 1, "end_line": 220}),
                ),
            ),
            ExplorationStep(
                key="rag_llm",
                label="查看 RAG / LLM 接入",
                module_hints=("rag", "llm", "deepseek", "retriever", "embedding"),
                tool_plan=(
                    ("ls", {"path": "rag", "recursive": False, "max_entries": 100}),
                    ("ls", {"path": "llm", "recursive": False, "max_entries": 100}),
                    ("read_file", {"path": "llm/deepseek_client.py", "start_line": 1, "end_line": 240}),
                ),
            ),
            ExplorationStep(
                key="tests",
                label="查看测试覆盖和验证方式",
                module_hints=("test_runtime.py", "test_stage3_server.py", "pytest", "compileall"),
                tool_plan=(
                    ("read_file", {"path": "test_runtime.py", "start_line": 1, "end_line": 160}),
                    ("read_file", {"path": "test_stage3_server.py", "start_line": 1, "end_line": 180}),
                ),
            ),
        ]

    @staticmethod
    def _generic_project_steps(target_path: str) -> List[ExplorationStep]:
        base = target_path.strip().strip("/").strip() or "."

        def join(path: str) -> str:
            if base == ".":
                return path
            if not path or path == ".":
                return base
            return f"{base.rstrip('/')}/{path.lstrip('/')}"

        return [
            ExplorationStep(
                key="root",
                label="查看目标目录结构",
                module_hints=(base.lower(), "目录:", "readme", "src", "test", "package", "pyproject", "go.mod", "cargo.toml"),
                tool_plan=(
                    ("ls", {"path": base, "recursive": False, "max_entries": 160}),
                    ("ls", {"path": base, "recursive": True, "max_entries": 220}),
                ),
            ),
            ExplorationStep(
                key="docs",
                label="查看 README / 项目说明",
                module_hints=("readme", "agent.md", "promote", "文档", "说明"),
                tool_plan=(
                    ("read_file", {"path": join("README.md"), "start_line": 1, "end_line": 220}),
                    ("read_file", {"path": join("AGENT.md"), "start_line": 1, "end_line": 160}),
                ),
            ),
            ExplorationStep(
                key="entry_config",
                label="查看入口与依赖配置",
                module_hints=("main.", "app.", "server.", "package.json", "pyproject", "requirements", "go.mod", "cargo.toml"),
                tool_plan=(
                    ("read_file", {"path": join("main.py"), "start_line": 1, "end_line": 220}),
                    ("read_file", {"path": join("package.json"), "start_line": 1, "end_line": 180}),
                    ("read_file", {"path": join("pyproject.toml"), "start_line": 1, "end_line": 180}),
                    ("read_file", {"path": join("requirements.txt"), "start_line": 1, "end_line": 160}),
                ),
            ),
            ExplorationStep(
                key="source",
                label="查看核心源码组织",
                module_hints=("src", "core", "lib", "internal", "pkg", "class ", "def ", "function "),
                tool_plan=(
                    ("ls", {"path": join("src"), "recursive": False, "max_entries": 140}),
                    ("ls", {"path": join("core"), "recursive": False, "max_entries": 140}),
                    ("grep", {"pattern": r"^(class|def|function|export|func) ", "path": base, "file_pattern": "*", "max_results": 80}),
                ),
            ),
            ExplorationStep(
                key="tests",
                label="查看测试与验证方式",
                module_hints=("test", "pytest", "unittest", "jest", "vitest", "cargo test", "go test"),
                tool_plan=(
                    ("ls", {"path": join("tests"), "recursive": False, "max_entries": 120}),
                    ("grep", {"pattern": r"(pytest|unittest|jest|vitest|cargo test|go test)", "path": base, "file_pattern": "*", "max_results": 80}),
                ),
            ),
        ]

    @property
    def coverage_ratio(self) -> float:
        if not self.steps:
            return 1.0
        return sum(1 for step in self.steps if step.done) / len(self.steps)

    @property
    def completed_steps(self) -> List[ExplorationStep]:
        return [step for step in self.steps if step.done]

    @property
    def unknown_steps(self) -> List[ExplorationStep]:
        return [step for step in self.steps if not step.done]

    def observe_tool(self, name: str, args: Any, content: str) -> InformationGain:
        before_modules = set(self.visited_modules)
        before_paths = set(self.visited_paths)
        before_symbols = set(self.visited_symbols)

        args = args if isinstance(args, dict) else {}
        path = args.get("path") or args.get("file_path")
        if path:
            self._add_path(str(path))
            self.attempted_tools.add(self._tool_key(name, args))

        text = str(content or "")
        informative = bool(text) and not self._is_low_information_result(text)
        if text:
            self.observations.append(text[:5000])
            for found in self._extract_paths(text):
                self._add_path(found)
            for symbol in self._extract_symbols(text):
                self.visited_symbols.add(symbol)

        if informative:
            for step in self.steps:
                if step.done:
                    continue
                if self._step_is_satisfied(step, name, args, text):
                    step.done = True
                    step.evidence.append(self._evidence_label(name, args, text))
                    self.visited_modules.add(step.key)

        new_modules = self.visited_modules - before_modules
        new_paths = self.visited_paths - before_paths if informative else set()
        new_symbols = self.visited_symbols - before_symbols if informative else set()
        gain = InformationGain(
            score=(len(new_modules) * 3) + len(new_paths) + min(3, len(new_symbols)),
            new_modules=sorted(new_modules),
            new_paths=sorted(new_paths)[:8],
            new_symbols=sorted(new_symbols)[:8],
        )
        self.low_gain_streak = self.low_gain_streak + 1 if gain.score <= 0 else 0
        return gain

    def ready_to_summarize(self) -> bool:
        enough_coverage = self.coverage_ratio >= self.COVERAGE_THRESHOLD and len(self.observations) >= self.MIN_EVIDENCE_ITEMS
        low_gain_done = self.low_gain_streak >= self.LOW_GAIN_LIMIT and len(self.observations) >= self.MIN_EVIDENCE_ITEMS
        no_next_step = not self.has_next_tool_call(("ls", "grep", "read_file")) and len(self.observations) >= 3
        return enough_coverage or low_gain_done or no_next_step

    def final_has_evidence(self, text: str) -> bool:
        return self._final_has_evidence(text)

    def should_allow_final(self, text: str) -> bool:
        if not str(text or "").strip():
            return False
        if self._looks_like_tool_placeholder(text):
            return False
        has_evidence = self._final_has_evidence(text)
        enough_coverage = self.coverage_ratio >= self.COVERAGE_THRESHOLD and len(self.observations) >= self.MIN_EVIDENCE_ITEMS
        low_gain_done = self.low_gain_streak >= self.LOW_GAIN_LIMIT and len(self.observations) >= self.MIN_EVIDENCE_ITEMS
        no_next_step = not self.has_next_tool_call(("ls", "grep", "read_file")) and len(self.observations) >= 3
        explicit_enough = self._model_claims_enough(text) and self.coverage_ratio >= 0.6 and len(self.observations) >= 4
        return has_evidence and (enough_coverage or low_gain_done or no_next_step or explicit_enough)

    def has_next_tool_call(self, available_tools: Iterable[str]) -> bool:
        return self._next_tool_call(available_tools, consume=False) is not None

    def next_tool_call(self, available_tools: Iterable[str]) -> Optional[Dict[str, Any]]:
        return self._next_tool_call(available_tools, consume=True)

    def calls_cover_planned_step(self, function_calls: Iterable[Dict[str, Any]]) -> bool:
        for call in function_calls or []:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or "")
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            requested_path = str(args.get("path") or args.get("file_path") or "").strip().replace("\\", "/")
            for step in self.unknown_steps:
                for planned_name, planned_args in step.tool_plan:
                    if name != planned_name:
                        continue
                    planned_path = str(planned_args.get("path") or planned_args.get("file_path") or "").strip().replace("\\", "/")
                    if name == "grep":
                        return True
                    if not planned_path or not requested_path:
                        return True
                    if requested_path == planned_path:
                        return True
                    if requested_path.startswith(planned_path.rstrip("/") + "/"):
                        return True
                    if planned_path.startswith(requested_path.rstrip("/") + "/"):
                        return True
        return False

    def _next_tool_call(self, available_tools: Iterable[str], consume: bool) -> Optional[Dict[str, Any]]:
        available = set(available_tools or [])
        for step in self.unknown_steps:
            for tool_name, args in step.tool_plan:
                if tool_name not in available:
                    continue
                key = self._tool_key(tool_name, args)
                if key in self.attempted_tools:
                    continue
                if consume:
                    self.attempted_tools.add(key)
                return {"name": tool_name, "args": dict(args)}
        return None

    def reflection_prompt(self, blocked_final: str = "") -> str:
        known = ", ".join(step.label for step in self.completed_steps) or "暂无充分覆盖"
        unknown = ", ".join(step.label for step in self.unknown_steps[:6]) or "核心 checklist 已覆盖"
        checklist = "\n".join(
            f"[{'x' if step.done else ' '}] {step.label}"
            for step in self.steps
        )
        next_step = self.unknown_steps[0].label if self.unknown_steps else "总结已有证据"
        blocked = ""
        if blocked_final:
            blocked = (
                "\n模型刚才尝试输出 Final Answer，但 Runtime 判定证据仍不足或缺少 Evidence。"
                "不要重复该答案，继续补齐未知区域。"
            )
        return (
            "【Project Exploration Reflection】\n"
            f"分析目标：{self.objective}\n"
            f"目标路径：{self.target_path}\n"
            f"你已经知道：{known}\n"
            f"仍然未知：{unknown}\n"
            f"Coverage: {self.coverage_ratio:.0%}; low_information_gain_streak={self.low_gain_streak}\n"
            f"Checklist:\n{checklist}\n"
            f"下一步优先探索：{next_step}\n"
            "请判断是否已经能够充分回答用户。若 checklist 未充分覆盖或答案缺少 Evidence，请继续调用 ls/grep/read_file；"
            "只有证据充分，并且最终回答能引用具体文件/模块时，才输出 Final Answer。"
            f"{blocked}"
        )

    def _step_is_satisfied(self, step: ExplorationStep, name: str, args: Dict[str, Any], text: str) -> bool:
        args = args if isinstance(args, dict) else {}
        raw_path = str(args.get("path") or args.get("file_path") or "").strip().replace("\\", "/")
        path_lower = raw_path.lower().strip("./")
        text_lower = str(text or "").lower()
        key = step.key

        def path_is(*prefixes: str) -> bool:
            return any(
                path_lower == prefix.strip("./").lower()
                or path_lower.startswith(prefix.strip("./").lower().rstrip("/") + "/")
                for prefix in prefixes
            )

        def file_is(*names: str) -> bool:
            base = os.path.basename(path_lower)
            return any(base == name.lower() for name in names)

        if key == "root":
            target = self.target_path.strip("./") or "."
            return name == "ls" and (raw_path in ("", ".", self.target_path) or path_lower == target.strip("./"))
        if key == "entry":
            return file_is("main.py", "agent.md", "promote.txt", "readme.md")
        if key == "runtime":
            return file_is("agent_runtime.py") or (name == "grep" and "function_call" in text_lower)
        if key == "tools":
            return path_is("tools/registry.py", "tools/file_tool.py", "tools/project_tools.py", "tools/base_tool.py")
        if key == "memory_context":
            return path_is("core/memory/store.py", "core/context/builder.py", "core/context/scratchpad.py", "core/context/compression.py")
        if key == "planning_todo":
            return path_is("core/planning/manager.py", "core/todo/models.py", "core/todo/store.py")
        if key == "permission":
            return path_is("core/permission/manager.py", "core/permission/web.py") or "web_permission" in path_lower
        if key == "server_web":
            return path_is("server/service.py", "server/app.py", "web/app.js", "web/styles.css", "web/index.html")
        if key == "rag_llm":
            return path_is("llm/deepseek_client.py", "rag/retriever.py", "rag/embedding.py")
        if key == "tests":
            return file_is("test_runtime.py", "test_stage3_server.py") or path_is("tests") or "pytest" in text_lower

        if key == "docs":
            return file_is("readme.md", "agent.md") or "readme" in path_lower
        if key == "entry_config":
            return file_is("main.py", "app.py", "server.py", "package.json", "pyproject.toml", "requirements.txt", "go.mod", "cargo.toml")
        if key == "source":
            return path_is("src", "core", "lib", "internal", "pkg") or bool(re.search(r"^\s*(class|def|function|export|func)\s+", text, re.MULTILINE))

        return any(hint.lower() in text_lower for hint in step.module_hints)

    def _final_has_evidence(self, text: str) -> bool:
        raw = str(text or "")
        lowered = raw.lower()
        has_label = any(marker in lowered for marker in ("evidence", "依据", "证据", "参考", "文件"))
        path_pattern = re.search(
            r"(\b[A-Za-z0-9_.-]+/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+|"
            r"\b[A-Za-z0-9_.-]+\.(?:py|md|txt|js|css|json|html|toml|yaml|yml))",
            raw,
        )
        mentions_observed = any(path.lower() in lowered for path in self.visited_paths if len(path) > 2)
        return has_label and bool(path_pattern or mentions_observed)

    @staticmethod
    def _model_claims_enough(text: str) -> bool:
        lowered = str(text or "").lower()
        return any(phrase in lowered for phrase in (
            "已经收集足够证据",
            "证据已经足够",
            "sufficient evidence",
            "enough evidence",
        ))

    @staticmethod
    def _looks_like_tool_placeholder(text: str) -> bool:
        lowered = str(text or "").strip().lower()
        return lowered.startswith(("[调用工具", "调用工具:", "read_file", "tool_calls", "<｜｜", "<|"))

    @staticmethod
    def _is_low_information_result(text: str) -> bool:
        stripped = str(text or "").strip()
        if not stripped:
            return True
        lowered = stripped.lower()
        low_markers = (
            "[错误]",
            "文件不存在",
            "路径不存在",
            "路径不是文件",
            "路径不是目录",
            "未找到匹配结果",
            "permission denied",
            "not found",
        )
        return any(marker in lowered for marker in low_markers)

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

    def _add_path(self, path: str) -> None:
        clean = path.strip().strip("\"'")
        if not clean:
            return
        clean = clean.replace("\\", "/")
        self.visited_paths.add(clean)
        if "." in os.path.basename(clean):
            self.visited_files.add(clean)

    @staticmethod
    def _extract_paths(text: str) -> List[str]:
        patterns = (
            r"文件:\s*([^\n]+)",
            r"目录:\s*([^\n]+)",
            r"(?:^|\n)\s*(core/[A-Za-z0-9_./-]+|tools/[A-Za-z0-9_./-]+|server/[A-Za-z0-9_./-]+|web/[A-Za-z0-9_./-]+|rag/[A-Za-z0-9_./-]+|llm/[A-Za-z0-9_./-]+)",
            r"(?:^|\n)\s*([A-Za-z0-9_.-]+\.(?:py|md|txt|js|css|json|html|toml|yaml|yml))",
        )
        found: List[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                value = match.group(1).strip().split("(", 1)[0].strip()
                if value and value not in found:
                    found.append(value)
        return found

    @staticmethod
    def _extract_symbols(text: str) -> List[str]:
        symbols = []
        for pattern in (r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)", r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)"):
            for match in re.finditer(pattern, text, flags=re.MULTILINE):
                symbols.append(match.group(1))
        return symbols[:30]

    @staticmethod
    def _tool_key(name: str, args: Dict[str, Any]) -> str:
        parts = [str(name or "")]
        for key in sorted(args):
            parts.append(f"{key}={args[key]}")
        return "|".join(parts)

    @staticmethod
    def _evidence_label(name: str, args: Dict[str, Any], content: str) -> str:
        path = args.get("path") or args.get("file_path") or ""
        if path:
            return f"{name}:{path}"
        first = str(content or "").splitlines()[0] if content else ""
        return f"{name}:{first[:80]}"
