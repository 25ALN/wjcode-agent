from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from core.todo import TodoList
from core.intent import contains_keyword

logger = logging.getLogger(__name__)


ACTION_KEYWORDS = (
    "实现",
    "完成",
    "修改",
    "修复",
    "重构",
    "新增",
    "添加",
    "接入",
    "集成",
    "优化",
    "测试",
    "运行",
    "执行",
    "检查",
    "调试",
    "定位",
    "复现",
    "删除",
    "创建",
    "编写",
    "继续",
    "fix",
    "fixing",
    "refactor",
    "implement",
    "integrate",
    "run",
    "runs",
    "running",
    "test",
    "tests",
    "testing",
    "debug",
    "debugging",
)

COMPLEX_KEYWORDS = (
    "全部",
    "阶段",
    "复杂",
    "规划",
    "多步",
    "多个",
    "一系列",
    "feature",
)

EXPLANATION_HINTS = (
    "是什么",
    "什么是",
    "为什么",
    "如何",
    "怎么",
    "怎样",
    "解释",
    "介绍",
    "说明",
    "概念",
    "原理",
    "区别",
    "优缺点",
    "你觉得",
    "吗",
    "是否",
    "能否",
    "可以",
    "what is",
    "what are",
    "why",
    "how",
    "explain",
    "describe",
    "concept",
    "difference",
    "should",
    "can",
    "could",
)

OBSERVATION_ERROR_KEYWORDS = (
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


@dataclass
class PlanState:
    objective: str
    steps: List[str] = field(default_factory=list)
    revision: int = 0
    mode: str = "react"
    last_observation: Optional[str] = None

    def format_for_prompt(self) -> str:
        if self.mode != "planning" or not self.steps:
            return ""
        lines = [f"模式: Planning + ReAct", f"目标: {self.objective}", f"计划版本: v{self.revision}"]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"  {i}. {step}")
        if self.last_observation:
            lines.append(f"最新观察: {self.last_observation}")
        return "\n".join(lines)


@dataclass
class PlanningUpdate:
    changed: bool
    reason: str
    plan: PlanState


class PlanningManager:

    def __init__(
        self,
        todo_list: Optional[TodoList] = None,
        min_complex_length: int = 32,
        enable_llm_planning: bool = False,
    ):
        self.todo_list = todo_list
        self.min_complex_length = min_complex_length
        self.enable_llm_planning = enable_llm_planning
        self.state = PlanState(objective="")

    def reset(self) -> None:
        """Clear the active plan without touching external history."""
        self.state = PlanState(objective="")

    def should_plan(self, user_input: str) -> bool:
        text = user_input.strip()
        if not text:
            return False

        lower = text.lower()
        has_action = contains_keyword(lower, ACTION_KEYWORDS)
        has_explanation = contains_keyword(lower, EXPLANATION_HINTS)

        # “复杂任务如何处理”这类是机制解释，不是需要拆解执行的复杂任务。
        if has_explanation and not has_action:
            return False

        if not has_action:
            return False

        if len(text) >= self.min_complex_length:
            return True

        return contains_keyword(lower, COMPLEX_KEYWORDS)

    def start_or_update_plan(
        self,
        user_input: str,
        llm_client: Optional[Any] = None,
        force: bool = False,
    ) -> PlanningUpdate:
        if not force and not self.should_plan(user_input):
            self.state.mode = "react"
            return PlanningUpdate(False, "任务较简单，使用纯 ReAct", self.state)

        steps = self._generate_steps(user_input, llm_client)
        self.state = PlanState(
            objective=user_input.strip(),
            steps=steps,
            revision=self.state.revision + 1,
            mode="planning",
        )
        self._sync_todo_with_plan(steps)
        return PlanningUpdate(True, "复杂任务已创建 Planning 计划", self.state)

    def observe_tool_result(self, tool_name: str, args: dict, observation: str) -> PlanningUpdate:
        if self.state.mode != "planning":
            return PlanningUpdate(False, "当前不是 Planning 模式", self.state)

        short_observation = self._shorten(observation, 240)
        self.state.last_observation = f"{tool_name}: {short_observation}"

        if self._is_error_observation(observation):
            changed = self._ensure_step("根据最新错误调整实现方案")
            changed = self._ensure_step("重新运行验证确认问题已解决") or changed
            if changed:
                self.state.revision += 1
                self._append_todo_if_missing("根据最新错误调整实现方案")
                self._append_todo_if_missing("重新运行验证确认问题已解决")
                return PlanningUpdate(True, "工具结果包含错误，已重规划", self.state)
            return PlanningUpdate(False, "错误处理步骤已存在", self.state)

        if tool_name in {"write_file", "edit_file"}:
            changed = self._ensure_step("运行测试或静态检查验证修改")
            if changed:
                self.state.revision += 1
                self._append_todo_if_missing("运行测试或静态检查验证修改")
                return PlanningUpdate(True, "代码已修改，补充验证步骤", self.state)

        return PlanningUpdate(False, "观察结果无需调整计划", self.state)

    def format_for_prompt(self) -> Optional[str]:
        text = self.state.format_for_prompt()
        return text or None

    def _generate_steps(self, user_input: str, llm_client: Optional[Any]) -> List[str]:
        if self.enable_llm_planning and llm_client is not None:
            try:
                steps = self._generate_steps_with_llm(user_input, llm_client)
                if steps:
                    return steps
            except Exception as exc:
                logger.warning(f"LLM 规划失败，回退到本地计划: {exc}")
        return self._fallback_steps(user_input)

    def _generate_steps_with_llm(self, user_input: str, llm_client: Any) -> List[str]:
        from core.message import Message

        prompt = (
            "请为以下编程任务生成 3 到 6 个可执行步骤。"
            "只输出每行一个步骤，不要编号，不要解释。\n\n"
            f"任务: {user_input}"
        )
        response = llm_client.generate(messages=[Message(role="user", content=prompt)])
        text = response if isinstance(response, str) else response.get("text", "")
        steps = [line.strip(" -0123456789.、") for line in text.splitlines()]
        return [step for step in steps if step][:6]

    @staticmethod
    def _fallback_steps(user_input: str) -> List[str]:
        text = user_input.lower()
        steps = ["分析相关项目结构和现有实现"]

        if any(word in text for word in ("bug", "修复", "错误", "失败", "报错", "fix")):
            steps.append("复现或定位问题根因")
        else:
            steps.append("设计实现方案并确定修改范围")

        steps.append("按计划修改代码或配置")
        steps.append("运行测试或命令验证结果")
        steps.append("根据验证结果修正计划并收尾总结")
        return steps

    def _sync_todo_with_plan(self, steps: List[str]) -> None:
        if self.todo_list is None:
            return
        if self.todo_list.count() == 0 or self.todo_list.is_all_done():
            self.todo_list.clear()
            self.todo_list.add_batch(steps)
            return
        for step in steps:
            self._append_todo_if_missing(step)

    def _append_todo_if_missing(self, description: str) -> bool:
        if self.todo_list is None:
            return False
        existing = {task.description for task in self.todo_list.list_all()}
        if description in existing:
            return False
        self.todo_list.add(description)
        return True

    def _ensure_step(self, description: str) -> bool:
        if description in self.state.steps:
            return False
        self.state.steps.append(description)
        return True

    @staticmethod
    def _is_error_observation(observation: str) -> bool:
        return any(keyword in observation for keyword in OBSERVATION_ERROR_KEYWORDS)

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        compact = " ".join(str(text).split())
        if len(compact) <= limit:
            return compact
        return compact[:limit] + "..."
