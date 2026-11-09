"""Sequential multi-agent orchestration built on top of :class:`AgentRuntime`.

Workers deliberately receive their own short memory and a restricted tool registry.
Only compact task results flow to the next worker.  This is the first safe stage of
multi-agent support: workers that can edit files are never run concurrently.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Generator, Iterable, List, Optional, TYPE_CHECKING

from core.events import (
    ASSISTANT_TEXT,
    DONE,
    ERROR,
    FINAL,
    PERMISSION_REQUEST,
    TOOL_CALL,
    TOOL_RESULT,
    AgentEvent,
    make_event,
)
from core.intent import contains_keyword
from core.memory import ShortMemory
from core.runtime.message import Message
from tools.registry import ToolRegistry

from .models import AgentRole, AgentTask, AgentTaskResult, TaskStatus

if TYPE_CHECKING:  # pragma: no cover
    from core.runtime.agent_runtime import AgentRuntime


AGENT_TASK_UPDATE = "agent_task_update"
AGENT_TASK_RESULT = "agent_task_result"

ACTION_KEYWORDS = (
    "修复", "修改", "实现", "新增", "添加", "删除", "重构", "接入", "集成",
    "测试", "运行", "执行", "检查", "调试", "定位", "复现", "编写",
    "fix", "implement", "refactor", "test", "debug", "run",
)

ROLE_TOOLS = {
    AgentRole.EXPLORE: ("ls", "grep", "read_file"),
    AgentRole.IMPLEMENT: ("ls", "grep", "read_file", "edit_file", "write_file"),
    AgentRole.TEST: ("ls", "grep", "read_file", "execute_code"),
    AgentRole.COORDINATOR: (),
}

ROLE_PROMPTS = {
    AgentRole.EXPLORE: (
        "你是 ExploreAgent，只负责只读代码探索和证据收集。"
        "只能根据实际工具结果下结论；不要修改文件、运行命令或猜测实现。"
        "最终输出应包含相关文件、调用链、根因假设和不确定项。"
    ),
    AgentRole.IMPLEMENT: (
        "你是 ImplementAgent，只在明确任务范围内实现修改。"
        "先阅读必要文件，再作最小改动；不要联网、删除文件或扩大修改范围。"
        "最终输出应列出修改文件、行为变化、未验证风险和建议验证命令。"
    ),
    AgentRole.TEST: (
        "你是 TestAgent，独立验证前序改动。"
        "只读取代码和执行测试/检查，不修改业务文件。"
        "最终输出必须明确通过、失败或阻塞，并列出实际运行命令和关键结果。"
    ),
    AgentRole.COORDINATOR: (
        "你是 CoordinatorAgent。根据用户需求和各 Worker 的结构化结果给出最终答复。"
        "只能把有证据或已验证的内容表述为结论；明确剩余风险和阻塞。"
        "不要声称自己执行了未出现在 Worker 结果中的操作。"
    ),
}


class MultiAgentOrchestrator:
    """Run Explore -> Implement -> Test and retain permission-resume state."""

    def __init__(self, parent_runtime: "AgentRuntime", max_retries: int = 2):
        self.parent_runtime = parent_runtime
        self.tasks: List[AgentTask] = []
        self._active_index = 0
        self._active_runtime: Optional["AgentRuntime"] = None
        self._active_input = ""
        self._pending_user_input = ""
        self.max_retries = max(0, int(max_retries))
        self._retry_count = 0

    @staticmethod
    def should_delegate(user_input: str, route=None) -> bool:
        return bool(route and route.mode == "planning" and route.task_kind == "implementation")

    @property
    def waiting_for_permission(self) -> bool:
        return self._active_runtime is not None and self.tasks and (
            self.tasks[self._active_index].status == TaskStatus.WAITING_PERMISSION
        )

    def run_events(self, user_input: str) -> Generator[AgentEvent, None, None]:
        self._pending_user_input = user_input
        self.tasks = [self._new_explore_task()]
        self._active_index = 0
        self._active_runtime = None
        self._retry_count = 0
        yield from self._run_from_active_task()

    def resume_events(self, approved: bool) -> Generator[AgentEvent, None, None]:
        if not self.waiting_for_permission or self._active_runtime is None:
            yield make_event(ERROR, {"message": "没有等待恢复的多 Agent 工具调用"}, self.parent_runtime.session_id)
            yield make_event(DONE, {}, self.parent_runtime.session_id)
            return
        task = self.tasks[self._active_index]
        task.status = TaskStatus.RUNNING
        yield self._task_event(task, "权限已确认，继续执行")
        paused = yield from self._consume_worker_events(task, self._active_runtime.resume_events(approved))
        if paused:
            return
        self._active_index += 1
        self._active_runtime = None
        if self._schedule_next_task(task):
            yield self._task_event(task, "已达到最大重试次数，Coordinator 结束重试")
        yield from self._run_from_active_task()

    def _run_from_active_task(self) -> Generator[AgentEvent, None, None]:
        while self._active_index < len(self.tasks):
            task = self.tasks[self._active_index]
            task.status = TaskStatus.RUNNING
            yield self._task_event(task, "开始执行")
            worker = self._make_worker(task)
            self._active_runtime = worker
            paused = yield from self._consume_worker_events(task, worker.run_events(self._task_input(task)))
            if paused:
                return
            self._active_index += 1
            self._active_runtime = None
            if self._schedule_next_task(task):
                yield self._task_event(task, "已达到最大重试次数，Coordinator 结束重试")
        yield from self._run_coordinator()

    def _consume_worker_events(
        self,
        task: AgentTask,
        events: Iterable[AgentEvent],
    ) -> Generator[AgentEvent, None, bool]:
        final_text = ""
        error_text = ""
        evidence: List[str] = []
        for event in events:
            if event.type == PERMISSION_REQUEST:
                task.status = TaskStatus.WAITING_PERMISSION
                yield self._task_event(task, "等待用户确认工具权限")
                payload = dict(event.data)
                payload["agent_role"] = task.role.value
                payload["agent_task_id"] = task.id
                yield make_event(PERMISSION_REQUEST, payload, self.parent_runtime.session_id)
                return True
            if event.type in {TOOL_CALL, TOOL_RESULT}:
                payload = dict(event.data)
                payload["agent_role"] = task.role.value
                payload["agent_task_id"] = task.id
                yield make_event(event.type, payload, self.parent_runtime.session_id)
                if event.type == TOOL_RESULT:
                    value = str(payload.get("result") or payload.get("content") or "").strip()
                    if value:
                        evidence.append(value[:500])
                continue
            if event.type == FINAL:
                final_text = str(event.data.get("content") or "").strip()
            elif event.type == ERROR:
                error_text = str(event.data.get("message") or "").strip()

        task.result = self._build_worker_result(task.role, final_text, evidence, error_text)
        task.status = task.result.status
        yield self._task_event(task, "执行完成")
        yield make_event(
            AGENT_TASK_RESULT,
            {"task": task.to_dict()},
            self.parent_runtime.session_id,
        )
        return False

    def _schedule_next_task(self, task: AgentTask) -> bool:
        """Let the coordinator extend the queue from one Worker Result."""
        result = task.result
        if result is None:
            return False

        if task.role == AgentRole.EXPLORE:
            if result.next_action == "implement" and result.status == TaskStatus.COMPLETED:
                self.tasks.append(self._new_implement_task())
            return False

        if task.role == AgentRole.IMPLEMENT:
            if result.next_action == "test" and result.status == TaskStatus.COMPLETED:
                self.tasks.append(self._new_test_task())
            return False

        if task.role == AgentRole.TEST and result.next_action == "retry_implement":
            if self._retry_count < self.max_retries:
                self._retry_count += 1
                self.tasks.append(self._new_implement_task(retry=True))
            else:
                result.next_action = "finish"
                result.risks.append(f"测试仍失败，已达到最大重试次数 ({self.max_retries})。")
                return True
        return False

    def _build_worker_result(
        self,
        role: AgentRole,
        final_text: str,
        tool_evidence: List[str],
        error_text: str,
    ) -> AgentTaskResult:
        text = (final_text or "Worker 未返回文本结论。").strip()
        status = TaskStatus.FAILED if error_text else self._status_from_text(role, text)
        summary = self._extract_field(text, ("摘要", "summary")) or text
        evidence = list(tool_evidence[-6:])
        declared_evidence = self._extract_field(text, ("证据", "evidence"))
        if declared_evidence:
            evidence.extend(
                item.strip() for item in re.split(r"[;；|]", declared_evidence) if item.strip()
            )
        next_action = self._next_action_for(role, status, text)
        return AgentTaskResult(
            status=status,
            summary=summary,
            evidence=self._unique(evidence)[:8],
            next_action=next_action,
            error=error_text or None,
        )

    @classmethod
    def _status_from_text(cls, role: AgentRole, text: str) -> TaskStatus:
        declared = cls._extract_field(text, ("状态", "status")).lower()
        if declared and any(word in declared for word in ("failed", "failure", "error", "失败", "阻塞", "blocked")):
            return TaskStatus.FAILED
        lower = text.lower()
        if role == AgentRole.TEST and (
            "测试失败" in text or "未通过" in text or "验证失败" in text
            or re.search(r"\b(failed|failure|error)\b", lower)
        ):
            return TaskStatus.FAILED
        lower = text.lower()
        return TaskStatus.COMPLETED

    @classmethod
    def _next_action_for(cls, role: AgentRole, status: TaskStatus, text: str) -> str:
        declared = cls._extract_field(text, ("下一步", "next_action", "next action")).lower()
        aliases = {
            "retry_implement": "retry_implement",
            "retry": "retry_implement",
            "重新实现": "retry_implement",
            "重新修改": "retry_implement",
            "implement": "implement",
            "implementation": "implement",
            "实现": "implement",
            "修改": "implement",
            "test": "test",
            "测试": "test",
            "finish": "finish",
            "结束": "finish",
            "完成": "finish",
        }
        requested = ""
        for key in sorted(aliases, key=len, reverse=True):
            if key in declared:
                requested = aliases[key]
                break

        if role == AgentRole.EXPLORE:
            no_change = ("无需修改", "不需要修改", "无需实现", "无需修复", "未发现需要修改")
            if status != TaskStatus.COMPLETED or requested == "finish" or any(item in text for item in no_change):
                return "finish"
            return "implement"
        if role == AgentRole.IMPLEMENT:
            return "test" if status == TaskStatus.COMPLETED else "finish"
        if role == AgentRole.TEST:
            return "retry_implement" if status == TaskStatus.FAILED else "finish"
        return "finish"

    @staticmethod
    def _extract_field(text: str, names: tuple[str, ...]) -> str:
        pattern = r"(?im)^\s*(?:" + "|".join(re.escape(name) for name in names) + r")\s*[:：]\s*(.+?)\s*$"
        match = re.search(pattern, text or "")
        return match.group(1).strip() if match else ""

    @staticmethod
    def _unique(items: Iterable[str]) -> List[str]:
        result: List[str] = []
        for item in items:
            value = str(item or "").strip()
            if value and value not in result:
                result.append(value)
        return result
    def _run_coordinator(self) -> Generator[AgentEvent, None, None]:
        task = AgentTask(
            role=AgentRole.COORDINATOR,
            instruction="汇总已经完成的任务结果并回复用户。",
        )
        task.status = TaskStatus.RUNNING
        yield self._task_event(task, "汇总 Worker 结果")
        worker = self._make_worker(task)
        final_text = ""
        error_text = ""
        for event in worker.run_events(self._coordinator_input()):
            if event.type == FINAL:
                final_text = str(event.data.get("content") or "").strip()
            elif event.type == ERROR:
                error_text = str(event.data.get("message") or "").strip()
        if error_text:
            task.status = TaskStatus.FAILED
            task.result = AgentTaskResult(status=TaskStatus.FAILED, error=error_text)
            yield self._task_event(task, "汇总失败")
            yield make_event(ERROR, {"message": error_text}, self.parent_runtime.session_id)
            yield make_event(DONE, {}, self.parent_runtime.session_id)
            return
        task.status = TaskStatus.COMPLETED
        task.result = AgentTaskResult(status=TaskStatus.COMPLETED, summary=final_text)
        yield self._task_event(task, "汇总完成")
        yield make_event(ASSISTANT_TEXT, {"content": final_text}, self.parent_runtime.session_id)
        yield make_event(FINAL, {"content": final_text}, self.parent_runtime.session_id)
        yield make_event(DONE, {}, self.parent_runtime.session_id)

    @staticmethod
    def _new_explore_task() -> AgentTask:
        return AgentTask(
            role=AgentRole.EXPLORE,
            instruction="定位与用户需求有关的代码、配置、调用链和现有测试。",
            acceptance_criteria=["给出可复核的文件或工具证据", "判断是否需要代码修改"],
        )

    @staticmethod
    def _new_implement_task(retry: bool = False) -> AgentTask:
        instruction = "根据探索证据完成用户要求的最小实现或修复。"
        if retry:
            instruction = "根据最近测试失败的证据修正前一次实现，并保持修改范围最小。"
        return AgentTask(
            role=AgentRole.IMPLEMENT,
            instruction=instruction,
            acceptance_criteria=["修改范围与需求一致", "说明变更和风险"],
        )

    @staticmethod
    def _new_test_task() -> AgentTask:
        return AgentTask(
            role=AgentRole.TEST,
            instruction="针对前序改动运行最相关的测试、检查或复现步骤。",
            acceptance_criteria=["报告实际验证命令", "明确通过、失败或阻塞"],
        )
    def _make_worker(self, task: AgentTask) -> "AgentRuntime":
        from core.runtime.agent_runtime import AgentRuntime

        registry = self._restricted_registry(task.role)
        parent = self.parent_runtime
        return AgentRuntime(
            llm_client=parent.llm,
            memory=ShortMemory(max_length=24, max_tokens=min(parent.max_conversation_tokens, 12000)),
            system_prompt=ROLE_PROMPTS[task.role] + "\n已有工作记录（仅作证据参考）：\n" + (parent._get_scratchpad_context() or ""),
            tool_registry=registry,
            permission_fn=parent.permission_fn,
            session_id=parent.session_id,
        )

    def _restricted_registry(self, role: AgentRole) -> Optional[ToolRegistry]:
        source = self.parent_runtime.tool_registry
        allowed = ROLE_TOOLS[role]
        if source is None or not allowed:
            return None
        registry = ToolRegistry()
        for name in allowed:
            tool = source.get(name)
            if tool is not None:
                registry.register(tool)
        return registry

    def _task_input(self, task: AgentTask) -> str:
        upstream = self._results_text()
        criteria = "\n".join(f"- {item}" for item in task.acceptance_criteria)
        return (
            f"【用户原始需求】\n{self._pending_user_input}\n\n"
            f"【你的子任务】\n{task.instruction}\n\n"
            f"【验收标准】\n{criteria}\n\n"
            f"【已完成 Worker 的结构化结果】\n{upstream or '暂无。'}\n\n"
            "【输出格式】保留自然语言说明，并分别写出：状态: completed/failed；摘要: ...；证据: ...；下一步: implement/test/retry_implement/finish。"
        )

    def _coordinator_input(self) -> str:
        return (
            f"【用户需求】\n{self._pending_user_input}\n\n"
            f"【Worker 结果】\n{self._results_text() or '没有可用 Worker 结果。'}\n\n"
            "请输出最终结论、实际完成内容、验证情况与剩余风险。"
        )

    def _results_text(self) -> str:
        parts = []
        for task in self.tasks:
            if task.result is None:
                continue
            result = task.result
            lines = [f"### {task.role.value}: {result.status.value}", result.summary]
            lines.append("下一步: " + result.next_action)
            if result.evidence:
                lines.append("证据: " + " | ".join(result.evidence[:4]))
            if result.error:
                lines.append("错误: " + result.error)
            parts.append("\n".join(line for line in lines if line))
        return "\n\n".join(parts)

    def _task_event(self, task: AgentTask, message: str) -> AgentEvent:
        return make_event(
            AGENT_TASK_UPDATE,
            {"task": task.to_dict(), "message": message},
            self.parent_runtime.session_id,
        )
