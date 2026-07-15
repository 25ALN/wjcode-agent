import logging
import re
from typing import List, Optional, Generator, Callable, Any, Dict
from uuid import uuid4
from core.message import Message
from core.memory import ShortMemory, LongMemory
from core.context import Scratchpad, build_context
from core.intent import contains_keyword, looks_like_file_reference
from core.events import (
    AgentEvent,
    ASSISTANT_TEXT,
    DONE,
    ERROR,
    FINAL,
    PERMISSION_REQUEST,
    PLANNING_UPDATE,
    TODO_UPDATE,
    TOOL_CALL,
    TOOL_RESULT,
    USER_MESSAGE,
    make_event,
)
from core.planning import PlanningManager
from core.runtime.exploration import ProjectExplorer
from core.web_permission import PermissionPending
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 12  # 复杂任务（如分析多文件目录）需要更多工具轮次

TOOL_ACTION_KEYWORDS = (
    "修复", "修改", "实现", "新增", "添加", "删除", "重构", "接入", "集成",
    "测试", "运行", "执行", "检查", "调试", "定位", "复现", "报错", "失败",
    "读取", "打开", "搜索", "查找", "grep", "diff",
    "补丁", "提交", "联网", "fix", "fixing", "implement", "refactor",
    "debug", "debugging", "test", "tests", "testing", "run", "runs", "running",
    "execute", "executing", "traceback", "exception", "failed", "commit", "open",
    "read", "search", "find",
)

TOOL_SOFT_ACTION_KEYWORDS = (
    "查看", "看一下", "分析", "inspect", "show", "view",
)

TOOL_RESOURCE_KEYWORDS = (
    "项目", "仓库", "目录", "文件", "源码", "函数", "类", "模块", "接口", "依赖",
    "配置", "日志", "页面", "前端", "后端", "服务", "路由", "数据库", "api",
    "网页", "网址", "url", "网站", "错误", "bug", "readme",
    "天气", "气温", "温度", "降雨", "下雨", "预报", "空气质量",
)

WEATHER_INFO_KEYWORDS = ("天气", "气温", "温度", "降雨", "下雨", "预报", "空气质量")
PROJECT_ANALYSIS_ACTION_KEYWORDS = (
    "分析", "梳理", "总结", "评估", "介绍", "亮点", "难点", "优点", "缺点",
    "架构", "结构", "overview", "review",
)
PROJECT_ANALYSIS_SCOPE_KEYWORDS = (
    "项目", "仓库", "目录", "代码库", "源码", "工程", "模块", "repo", "repository",
)
PROJECT_ANALYSIS_MUTATION_KEYWORDS = (
    "修改", "修复", "新增", "添加", "删除", "重构", "写入", "编辑",
    "运行", "测试", "调试", "复现", "执行",
)
PROJECT_ANALYSIS_READ_TOOLS = ("ls", "grep", "read_file")
FRESH_INFO_TIME_KEYWORDS = (
    "今天", "明天", "后天", "上午", "下午", "今晚", "早上", "中午", "晚上",
    "本周", "周一", "周二", "周三", "周四", "周五", "周六", "周日",
    "实时", "最新", "当前", "现在",
)
LOCATION_HINT_KEYWORDS = ("省", "市", "区", "县", "镇", "街道", "贵州", "贵阳", "南明")
EXTERNAL_SEARCH_CONTINUATION_HINTS = (
    "查吧", "查一下", "查询一下", "搜吧", "搜一下", "搜索一下", "那就查", "帮我查",
)

TOOL_INTENT_KEYWORDS = TOOL_ACTION_KEYWORDS + TOOL_SOFT_ACTION_KEYWORDS + TOOL_RESOURCE_KEYWORDS

DIRECT_ANSWER_HINTS = (
    "你觉得", "是什么", "什么是", "为什么", "如何", "怎么", "怎样", "解释",
    "介绍", "说明", "概念", "原理", "关键", "原则", "建议", "区别",
    "优缺点", "有办法", "可以", "能否", "是否", "吗", "哪里", "哪儿",
    "在哪", "在哪里", "从哪里", "存储在哪里", "保存在哪里", "怎么样", "怎么样了",
    "状态", "现状", "情况", "目前", "最近", "恢复", "what is",
    "what are", "why", "how", "explain", "describe", "concept", "difference",
    "should", "can", "could",
)

PSEUDO_TOOL_MARKERS = (
    "DSML",
    "tool_calls",
    "tool_call",
    "<｜｜",
    "<|",
    "[调用工具",
    "调用工具:",
)

CONTINUATION_TOOL_HINTS = ("继续", "接着", "刚才", "上一步", "没完成", "未完成")


class AgentRuntime:
    """Agent 运行时调度器

    负责协调 memory、RAG、Tool、LLM 四大模块，完成一次完整的对话循环。
    支持 Function Calling（与 LLM 后端无关，DeepSeek/Gemini 均可）。

    """

    def __init__(
        self,
        llm_client: Any,
        memory: Optional[ShortMemory] = None,
        long_memory: Optional[LongMemory] = None,
        system_prompt: Optional[str] = None,
        rag_fn: Optional[Callable[[str], List[str]]] = None,
        tool_fn: Optional[Callable[[str], str]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_conversation_tokens: int = 100000,  # DeepSeek 128K 上下文，留足空间
        auto_summarize_threshold: int = 8,       # 每8轮自动总结一次
        project_context: Optional[str] = None,
        todo_list: Optional[object] = None,
        planning_manager: Optional[PlanningManager] = None,
        permission_fn: Optional[Callable] = None,
        context_compressor: Optional[object] = None,
        scratchpad: Optional[Scratchpad] = None,
        session_id: Optional[str] = None,
    ):
        self.llm = llm_client
        self.memory = memory or ShortMemory(max_length=60, max_tokens=max_conversation_tokens)
        self.long_memory = long_memory
        self.system_prompt = system_prompt or "你是一个有用的AI助手。"
        self.rag_fn = rag_fn
        self.tool_fn = tool_fn
        self.tool_registry = tool_registry
        self.max_conversation_tokens = max_conversation_tokens
        self.auto_summarize_threshold = auto_summarize_threshold
        self._conversation_turns = 0
        self.project_context = project_context
        self.todo_list = todo_list
        self.planning_manager = planning_manager
        self.permission_fn = permission_fn
        self.context_compressor = context_compressor
        self.scratchpad = scratchpad or Scratchpad()
        self.session_id = session_id or uuid4().hex
        self._pending_tool_call: Optional[dict] = None
        self._pending_function_calls: Optional[List[dict]] = None
        self._pending_function_index = 0
        self._pending_plan_changed = False
        self._pending_context: Optional[List[Message]] = None
        self._pending_tools: Optional[List[Dict[str, Any]]] = None
        self._pending_tool_round = 0
        self._pending_text: Optional[str] = None

    def run(self, user_input: str) -> str:
        user_message = Message(role="user", content=user_input)
        self.memory.add_message(user_message)
        self._start_scratchpad(user_input)

        if self.context_compressor is not None:
            self._check_and_compress()

        if self.planning_manager is not None:
            self._maybe_start_plan(user_input)

        if self._should_expose_tools(user_input):
            return self._run_with_native_tools(user_input)

        return self._run_with_callback_tool()

    def run_events(self, user_input: str) -> Generator[AgentEvent, None, None]:
        """Run the agent and emit structured events for Web/API layers."""
        if self._pending_tool_call is not None:
            yield self._event(ERROR, {
                "message": "当前有等待权限确认的工具调用，请先调用 resume_events()",
            })
            yield self._event(DONE, {})
            return

        try:
            user_message = Message(role="user", content=user_input)
            self.memory.add_message(user_message)
            self._start_scratchpad(user_input)
            yield self._event(USER_MESSAGE, {"content": user_input})

            if self.context_compressor is not None:
                self._check_and_compress()

            if self.planning_manager is not None:
                update = self._maybe_start_plan(user_input)
                if update is not None and update.changed:
                    yield self._event(PLANNING_UPDATE, self._planning_payload(update))
                    yield self._event(TODO_UPDATE, self._todo_payload())

            if self._should_expose_tools(user_input):
                yield from self._run_native_tools_events(user_input)
                return

            yield self._event(ASSISTANT_TEXT, {"content": "", "phase": "model_start"})
            reply = self._run_with_callback_tool()
            yield self._event(FINAL, {"content": reply})
            yield self._event(DONE, {})
        except Exception as exc:
            logger.exception("Agent event run failed")
            yield self._event(ERROR, {"message": f"Agent 执行失败: {str(exc)[:200]}"})
            yield self._event(DONE, {})

    def _should_expose_tools(self, user_input: str) -> bool:
        """Decide whether this turn should expose tool schemas to the model.

        General advice/explanation questions should remain plain chat even when
        a registry exists. Exposing file/code tools on every turn makes models
        over-eager to call read_file/grep for conceptual questions and can leave
        the Web UI waiting on unnecessary tool work.
        """
        if not self.tool_registry or self.tool_registry.tool_count <= 0:
            return False

        text = str(user_input or "").strip()
        if not text:
            return False
        lower = text.lower()

        for name in self.tool_registry.list_names():
            if name and re.search(rf"(?<![\w-]){re.escape(name.lower())}(?![\w-])", lower):
                return True

        if self._requires_fresh_external_tool(lower):
            return True
        if self._continues_fresh_external_request(lower):
            return True
        if self._is_project_analysis_request(lower):
            return bool(self._registered_project_read_tools())

        has_direct_answer_hint = contains_keyword(lower, DIRECT_ANSWER_HINTS)
        has_hard_action = contains_keyword(lower, TOOL_ACTION_KEYWORDS)
        has_soft_action = contains_keyword(lower, TOOL_SOFT_ACTION_KEYWORDS)
        has_resource_hint = contains_keyword(lower, TOOL_RESOURCE_KEYWORDS) or looks_like_file_reference(lower)
        has_action_intent = has_hard_action or (has_soft_action and has_resource_hint)

        if has_direct_answer_hint and not self._is_explicit_tool_request(
            lower,
            has_hard_action=has_hard_action,
            has_soft_action=has_soft_action,
            has_resource_hint=has_resource_hint,
        ):
            return False

        if has_action_intent:
            return True

        if contains_keyword(lower, CONTINUATION_TOOL_HINTS):
            state = getattr(self.planning_manager, "state", None)
            if state is not None and getattr(state, "mode", "react") == "planning":
                return True

        return has_resource_hint and not has_direct_answer_hint

    def _select_native_tool_declarations(self, user_input: str) -> List[Dict[str, Any]]:
        if not self.tool_registry:
            return []
        names = self._tools_for_turn(user_input)
        return self.tool_registry.get_function_declarations(names=names)

    def _tools_for_turn(self, user_input: str) -> Optional[List[str]]:
        text = str(user_input or "").strip().lower()
        if not text or not self.tool_registry:
            return None

        if self._requires_fresh_external_tool(text) or self._continues_fresh_external_request(text):
            return ["web_search"] if self._has_registered_tool("web_search") else []

        if self._is_project_analysis_request(text):
            return self._registered_project_read_tools()

        if self._has_registered_tool("execute_code") and self._needs_code_execution(text):
            tools = []
            if self._has_registered_tool("read_file"):
                tools.append("read_file")
            if self._has_registered_tool("ls"):
                tools.append("ls")
            if self._has_registered_tool("grep"):
                tools.append("grep")
            if self._has_registered_tool("edit_file"):
                tools.append("edit_file")
            if self._has_registered_tool("write_file"):
                tools.append("write_file")
            if self._has_registered_tool("execute_code"):
                tools.append("execute_code")
            if self._has_registered_tool("web_search") and contains_keyword(text, WEATHER_INFO_KEYWORDS):
                tools.append("web_search")
            return tools or None

        return None

    def _needs_code_execution(self, lower_text: str) -> bool:
        code_hints = ("运行", "执行", "测试", "调试", "复现", "报错", "失败", "shell", "python", "代码")
        return contains_keyword(lower_text, code_hints)

    def _is_project_analysis_request(self, lower_text: str) -> bool:
        """Detect read-only project/dir analysis turns.

        These turns need repository evidence, but they should not expose write,
        execution, or network tools. Keeping the tool surface small prevents the
        model from wandering into repeated file reads or irrelevant actions.
        """
        if not lower_text:
            return False
        if self._requires_fresh_external_tool(lower_text):
            return False
        if contains_keyword(lower_text, PROJECT_ANALYSIS_MUTATION_KEYWORDS):
            return False

        has_scope = (
            contains_keyword(lower_text, PROJECT_ANALYSIS_SCOPE_KEYWORDS)
            or looks_like_file_reference(lower_text)
        )
        has_analysis_intent = contains_keyword(lower_text, PROJECT_ANALYSIS_ACTION_KEYWORDS)
        return has_scope and has_analysis_intent

    def _registered_project_read_tools(self) -> List[str]:
        return [
            name for name in PROJECT_ANALYSIS_READ_TOOLS
            if self._has_registered_tool(name)
        ]

    def _has_registered_tool(self, name: str) -> bool:
        return bool(self.tool_registry and name in self.tool_registry.list_names())

    def _requires_fresh_external_tool(self, lower_text: str) -> bool:
        if not self._has_registered_tool("web_search"):
            return False
        if contains_keyword(lower_text, WEATHER_INFO_KEYWORDS):
            return (
                contains_keyword(lower_text, FRESH_INFO_TIME_KEYWORDS)
                or contains_keyword(lower_text, LOCATION_HINT_KEYWORDS)
                or contains_keyword(lower_text, ("查询", "搜索", "查一下", "搜一下", "联网"))
            )
        return False

    def _continues_fresh_external_request(self, lower_text: str) -> bool:
        if not self._has_registered_tool("web_search"):
            return False
        if not contains_keyword(lower_text, EXTERNAL_SEARCH_CONTINUATION_HINTS):
            return False

        recent_messages = self.memory.messages[:-1] if self.memory.messages else []
        recent_text = "\n".join(msg.content for msg in recent_messages[-6:]).lower()
        return contains_keyword(
            recent_text,
            WEATHER_INFO_KEYWORDS + ("联网", "搜索", "查询", "实时", "最新", "工具模式"),
        )

    def _is_explicit_tool_request(
        self,
        lower_text: str,
        has_hard_action: bool,
        has_soft_action: bool,
        has_resource_hint: bool,
    ) -> bool:
        """Return True only when a direct-question turn still asks us to act.

        Questions like “长期记忆存在哪里、打开历史从哪里恢复” contain words
        such as “打开” but are asking for an explanation, not file inspection.
        """
        has_action = has_hard_action or has_soft_action
        if not has_action:
            return False
        if looks_like_file_reference(lower_text):
            return True

        location_question_hints = (
            "哪里",
            "哪儿",
            "在哪",
            "在哪里",
            "从哪里",
            "存储在哪里",
            "保存在哪里",
        )
        internal_mechanism_terms = (
            "记忆",
            "memory",
            "历史",
            "历史记录",
            "会话",
            "上下文",
            "context",
            "planning",
            "todo",
            "scratchpad",
        )
        if (
            contains_keyword(lower_text, location_question_hints)
            and contains_keyword(lower_text, internal_mechanism_terms)
            and not looks_like_file_reference(lower_text)
            and not contains_keyword(lower_text, TOOL_RESOURCE_KEYWORDS)
        ):
            return False

        imperative_prefixes = (
            "检查",
            "查看",
            "读取",
            "打开",
            "搜索",
            "查找",
            "运行",
            "测试",
            "修改",
            "修复",
            "实现",
            "分析",
            "调试",
            "定位",
            "复现",
            "新增",
            "添加",
            "删除",
            "重构",
        )
        request_prefixes = (
            "请你",
            "请帮我",
            "请帮忙",
            "帮我",
            "帮忙",
            "麻烦你",
            "麻烦",
            "需要你",
            "你需要",
            "给我",
            "能不能",
            "能否",
            "可以帮我",
            "可不可以",
        )
        candidates = [lower_text]
        stripped = lower_text.lstrip()
        for prefix in request_prefixes:
            if stripped.startswith(prefix):
                candidates.append(stripped[len(prefix):].lstrip(" ，,。:："))
                break

        return any(
            candidate.startswith(imperative_prefixes) and (has_resource_hint or has_hard_action)
            for candidate in candidates
        )

    def resume_events(self, approved: bool) -> Generator[AgentEvent, None, None]:
        """Resume a paused Web/API run after a permission decision."""
        if self._pending_tool_call is None:
            yield self._event(ERROR, {"message": "没有等待恢复的工具调用"})
            return

        context = self._pending_context
        tools = self._pending_tools
        function_call = self._pending_tool_call
        function_calls = self._pending_function_calls or ([function_call] if function_call else [])
        function_index = self._pending_function_index
        plan_changed = self._pending_plan_changed
        tool_round = self._pending_tool_round
        pending_text = self._pending_text

        self._pending_context = None
        self._pending_tools = None
        self._pending_tool_call = None
        self._pending_function_calls = None
        self._pending_function_index = 0
        self._pending_plan_changed = False
        self._pending_tool_round = 0
        self._pending_text = None

        if context is None or tools is None:
            yield self._event(ERROR, {"message": "等待恢复的上下文已丢失"})
            return

        plan_changed = yield from self._process_tool_call_batch_events(
            context=context,
            tools=tools,
            function_calls=function_calls,
            start_index=function_index,
            tool_round=tool_round,
            pending_text=pending_text,
            plan_changed=plan_changed,
            first_permission_override=bool(approved),
            emit_start_tool_call=False,
        )
        if plan_changed is None:
            return

        self._append_batch_context_updates(context, plan_changed)
        if plan_changed:
            yield self._event(PLANNING_UPDATE, self._current_planning_payload())
            yield self._event(TODO_UPDATE, self._todo_payload())

        yield from self._continue_native_tools_events(context, tools, tool_round, pending_text)

    def _run_with_native_tools(self, user_input: str) -> str:
        rag_results = self._get_rag()
        long_memory_context = self._get_long_memory_context()
        todo_context = self._get_todo_context()
        planning_context = self._get_planning_context()
        scratchpad_context = self._get_scratchpad_context()

        context = self._build_context(
            rag_results=rag_results,
            long_memory_context=long_memory_context,
            todo_context=todo_context,
            planning_context=planning_context,
            scratchpad_context=scratchpad_context,
        )
        self._append_tool_workflow_context(context, user_input)

        tools = self._select_native_tool_declarations(user_input)
        project_analysis = self._is_project_analysis_request(str(user_input or "").strip().lower())
        explorer = ProjectExplorer(user_input) if project_analysis else None
        if explorer:
            self._append_project_exploration_reflection(context, explorer)
        tool_round = 0
        pending_text = None  # 缓存的文本回复（用于 text + function_call 共存场景）

        while project_analysis or tool_round < MAX_TOOL_ROUNDS:
            tool_round += 1

            response = self._normalize_llm_response(
                self.llm.generate(messages=context, tools=tools)
            )
            text = response.get("text")
            function_calls = self._ensure_function_call_ids(
                self._extract_function_calls(response),
                tool_round,
            )
            if project_analysis and explorer and function_calls and explorer.ready_to_summarize():
                return self._force_final_answer_from_tool_context(context, user_input, explorer)
            if (
                project_analysis
                and explorer
                and function_calls
                and not explorer.calls_cover_planned_step(function_calls)
            ):
                if self._run_project_explorer_auto_step(
                    context, explorer, user_input, tool_round, blocked_final="模型请求的工具调用未覆盖当前 exploration checklist。"
                ):
                    continue

            # 纯工具草稿（无 function_call）→ 项目探索自动补证据，其他场景纠偏重试。
            if text and not function_calls and self._is_unusable_tool_draft(text):
                if project_analysis and explorer:
                    if self._run_project_explorer_auto_step(context, explorer, user_input, tool_round, blocked_final=text):
                        continue
                    return self._force_final_answer_from_tool_context(context, user_input, explorer)
                context.append(self._native_tool_retry_message())
                continue

            #纯文本（无 function_call）→ 项目分析需先满足 coverage/evidence，其余立即返回。
            if text and not function_calls:
                if project_analysis and explorer and not explorer.should_allow_final(text):
                    if self._run_project_explorer_auto_step(context, explorer, user_input, tool_round, blocked_final=text):
                        continue
                    return self._force_final_answer_from_tool_context(context, user_input, explorer)
                return self._finalize_response(text)

            #text + function_call 共存 → 缓存文本，继续执行工具
            if text and function_calls:
                pending_text = text
                logger.debug(f"LLM 同时返回文本和工具调用，缓存文本: {text[:80]}...")

            #纯 function_call → 执行工具
            if function_calls:
                call_names = [call.get("name", "") for call in function_calls]
                fn_call_msg = Message(
                    role="assistant",
                    content=f"[调用工具: {', '.join(call_names)}]",
                    metadata={
                        "function_call": function_calls[0],
                        "function_calls": function_calls,
                    },
                )
                self.memory.add_message(fn_call_msg)
                context.append(fn_call_msg)

                plan_changed = False

                for function_call in function_calls:
                    fn_name = function_call.get("name", "")
                    fn_args = function_call.get("args", {})

                    round_label = str(tool_round) if project_analysis else f"{tool_round}/{MAX_TOOL_ROUNDS}"
                    logger.info(f"LLM 请求调用 Tool ({round_label}): "
                                f"{fn_name}({list(fn_args.keys()) if isinstance(fn_args, dict) else 'invalid_args'})")

                    tool_result = self._execute_tool(fn_name, fn_args)
                    tool_content = self._truncate_tool_result(tool_result)

                    tool_msg = Message(
                        role="tool",
                        content=tool_content,
                        name=fn_name,
                        metadata={
                            "function_response": {
                                "id": function_call.get("id"),
                                "name": fn_name,
                                "response": {"result": tool_content},
                            }
                        },
                    )
                    self.memory.add_message(tool_msg)
                    context.append(tool_msg)
                    self._observe_scratchpad(fn_name, fn_args, tool_content)
                    if explorer and fn_name in PROJECT_ANALYSIS_READ_TOOLS:
                        explorer.observe_tool(fn_name, fn_args, tool_content)

                    if self.planning_manager is not None:
                        planning_update = self._update_plan_from_observation(
                            fn_name,
                            fn_args,
                            tool_content,
                        )
                        if planning_update is not None and planning_update.changed:
                            plan_changed = True

                refreshed_scratchpad = self._get_scratchpad_context()
                if refreshed_scratchpad:
                    context.append(self._scratchpad_context_message(refreshed_scratchpad))

                if plan_changed:
                    refreshed_plan = self._get_planning_context()
                    if refreshed_plan:
                        context.append(Message(
                            role="system",
                            content=(
                                "【当前执行计划已更新】\n"
                                f"{refreshed_plan}\n"
                                "请依据更新后的计划继续执行。"
                            ),
                        ))
                if explorer:
                    self._append_project_exploration_reflection(context, explorer)
                continue

            # 既没有 text 也没有 function_call — 异常
            if pending_text:
                return self._finalize_response(pending_text)
            return "[Agent 未返回有效响应]"

        # 循环耗尽 — 如果有缓存文本，返回它；否则报错
        if explorer:
            return self._force_final_answer_from_tool_context(context, user_input, explorer)
        if pending_text:
            logger.info(f"工具轮次用尽（{MAX_TOOL_ROUNDS}），返回缓存的文本")
            return self._finalize_response(pending_text)

        return f"[Agent 调用工具次数过多（{MAX_TOOL_ROUNDS}次），已中止]"

    def _run_native_tools_events(self, user_input: str) -> Generator[AgentEvent, None, None]:
        rag_results = self._get_rag()
        long_memory_context = self._get_long_memory_context()
        todo_context = self._get_todo_context()
        planning_context = self._get_planning_context()
        scratchpad_context = self._get_scratchpad_context()

        context = self._build_context(
            rag_results=rag_results,
            long_memory_context=long_memory_context,
            todo_context=todo_context,
            planning_context=planning_context,
            scratchpad_context=scratchpad_context,
        )
        self._append_tool_workflow_context(context, user_input)
        tools = self._select_native_tool_declarations(user_input)
        yield from self._continue_native_tools_events(context, tools, 0, None, user_input=user_input)

    def _continue_native_tools_events(
        self,
        context: List[Message],
        tools: List[Dict[str, Any]],
        tool_round: int,
        pending_text: Optional[str],
        user_input: Optional[str] = None,
    ) -> Generator[AgentEvent, None, None]:
        if user_input is None:
            user_input = self._last_user_content(context)
        project_analysis = self._is_project_analysis_request(str(user_input or "").strip().lower())
        explorer = ProjectExplorer(user_input) if project_analysis else None
        if explorer:
            self._rebuild_explorer_from_context(explorer, context)
            self._append_project_exploration_reflection(context, explorer)
        while project_analysis or tool_round < MAX_TOOL_ROUNDS:
            tool_round += 1

            response = self._normalize_llm_response(
                self.llm.generate(messages=context, tools=tools)
            )
            text = response.get("text")
            function_calls = self._ensure_function_call_ids(
                self._extract_function_calls(response),
                tool_round,
            )
            if project_analysis and explorer and function_calls and explorer.ready_to_summarize():
                reply = self._force_final_answer_from_tool_context(context, user_input, explorer)
                yield self._event(ASSISTANT_TEXT, {"content": reply})
                yield self._event(FINAL, {"content": reply})
                yield self._event(DONE, {})
                return
            if (
                project_analysis
                and explorer
                and function_calls
                and not explorer.calls_cover_planned_step(function_calls)
            ):
                did_step = yield from self._run_project_explorer_auto_step_events(
                    context, tools, explorer, user_input, tool_round,
                    blocked_final="模型请求的工具调用未覆盖当前 exploration checklist。",
                )
                if did_step:
                    continue

            if text and not function_calls and self._is_unusable_tool_draft(text):
                if project_analysis and explorer:
                    did_step = yield from self._run_project_explorer_auto_step_events(
                        context, tools, explorer, user_input, tool_round, blocked_final=text
                    )
                    if did_step:
                        continue
                    reply = self._force_final_answer_from_tool_context(context, user_input, explorer)
                    yield self._event(ASSISTANT_TEXT, {"content": reply})
                    yield self._event(FINAL, {"content": reply})
                    yield self._event(DONE, {})
                    return
                context.append(self._native_tool_retry_message())
                continue

            if text and not function_calls:
                if project_analysis and explorer and not explorer.should_allow_final(text):
                    did_step = yield from self._run_project_explorer_auto_step_events(
                        context, tools, explorer, user_input, tool_round, blocked_final=text
                    )
                    if did_step:
                        continue
                    reply = self._force_final_answer_from_tool_context(context, user_input, explorer)
                    yield self._event(ASSISTANT_TEXT, {"content": reply})
                    yield self._event(FINAL, {"content": reply})
                    yield self._event(DONE, {})
                    return
                if text and not self._is_unusable_tool_draft(text):
                    yield self._event(ASSISTANT_TEXT, {"content": text})
                reply = self._finalize_response(text)
                yield self._event(FINAL, {"content": reply})
                yield self._event(DONE, {})
                return

            if text and not self._is_unusable_tool_draft(text):
                yield self._event(ASSISTANT_TEXT, {"content": text})

            if text and function_calls:
                pending_text = text

            if function_calls:
                call_names = [call.get("name", "") for call in function_calls]
                fn_call_msg = Message(
                    role="assistant",
                    content=f"[调用工具: {', '.join(call_names)}]",
                    metadata={
                        "function_call": function_calls[0],
                        "function_calls": function_calls,
                    },
                )
                self.memory.add_message(fn_call_msg)
                context.append(fn_call_msg)

                plan_changed = yield from self._process_tool_call_batch_events(
                    context=context,
                    tools=tools,
                    function_calls=function_calls,
                    start_index=0,
                    tool_round=tool_round,
                    pending_text=pending_text,
                    plan_changed=False,
                )
                if plan_changed is None:
                    return

                if explorer:
                    self._observe_explorer_from_function_calls(explorer, context, function_calls)
                self._append_batch_context_updates(context, plan_changed)
                if plan_changed:
                    yield self._event(PLANNING_UPDATE, self._current_planning_payload())
                    yield self._event(TODO_UPDATE, self._todo_payload())
                if explorer:
                    self._append_project_exploration_reflection(context, explorer)
                continue

            if pending_text:
                reply = self._finalize_response(pending_text)
                yield self._event(FINAL, {"content": reply})
                yield self._event(DONE, {})
                return

            yield self._event(ERROR, {"message": "Agent 未返回有效响应"})
            return

        if explorer:
            reply = self._force_final_answer_from_tool_context(context, user_input, explorer)
            yield self._event(ASSISTANT_TEXT, {"content": reply})
            yield self._event(FINAL, {"content": reply})
            yield self._event(DONE, {})
            return

        if pending_text:
            reply = self._finalize_response(pending_text)
            yield self._event(FINAL, {"content": reply})
            yield self._event(DONE, {})
            return

        yield self._event(ERROR, {
            "message": f"Agent 调用工具次数过多（{MAX_TOOL_ROUNDS}次），已中止",
        })

    def _process_tool_call_batch_events(
        self,
        context: List[Message],
        tools: List[Dict[str, Any]],
        function_calls: List[dict],
        start_index: int,
        tool_round: int,
        pending_text: Optional[str],
        plan_changed: bool = False,
        first_permission_override: Optional[bool] = None,
        emit_start_tool_call: bool = True,
    ) -> Generator[AgentEvent, None, Optional[bool]]:
        """Execute pending tool calls from one assistant tool_calls message.

        OpenAI/DeepSeek require every assistant message with tool_calls to be
        followed by one tool message for each tool_call_id before the next model
        request. If permission pauses in the middle of a batch, resume must
        finish the same batch instead of immediately calling the model again.
        """
        if not function_calls:
            return plan_changed

        for index in range(start_index, len(function_calls)):
            function_call = function_calls[index]
            if index == start_index and first_permission_override is not None:
                permission_override = first_permission_override
                emit_tool_call = emit_start_tool_call
            else:
                permission_override = None
                emit_tool_call = True

            if emit_tool_call:
                yield self._event(TOOL_CALL, self._tool_call_payload(function_call))

            try:
                tool_msg, changed = self._complete_tool_call(function_call, permission_override)
            except PermissionPending as pending:
                self._store_pending_tool_batch(
                    context=context,
                    tools=tools,
                    function_calls=function_calls,
                    function_index=index,
                    tool_round=tool_round,
                    pending_text=pending_text,
                    plan_changed=plan_changed,
                )
                yield self._event(PERMISSION_REQUEST, pending.request.to_dict())
                return None

            self.memory.add_message(tool_msg)
            context.append(tool_msg)
            plan_changed = changed or plan_changed
            yield self._event(TOOL_RESULT, self._tool_result_payload(tool_msg))

        return plan_changed

    def _store_pending_tool_batch(
        self,
        context: List[Message],
        tools: List[Dict[str, Any]],
        function_calls: List[dict],
        function_index: int,
        tool_round: int,
        pending_text: Optional[str],
        plan_changed: bool,
    ) -> None:
        function_call = function_calls[function_index]
        self._pending_tool_call = function_call
        self._pending_function_calls = function_calls
        self._pending_function_index = function_index
        self._pending_plan_changed = plan_changed
        self._pending_context = context
        self._pending_tools = tools
        self._pending_tool_round = tool_round
        self._pending_text = pending_text

    def _append_batch_context_updates(self, context: List[Message], plan_changed: bool) -> None:
        refreshed_scratchpad = self._get_scratchpad_context()
        if refreshed_scratchpad:
            context.append(self._scratchpad_context_message(refreshed_scratchpad))

        if plan_changed:
            refreshed_plan = self._get_planning_context()
            if refreshed_plan:
                context.append(self._planning_context_message(refreshed_plan))

    def _run_project_explorer_auto_step(
        self,
        context: List[Message],
        explorer: ProjectExplorer,
        user_input: str,
        tool_round: int,
        blocked_final: str = "",
    ) -> bool:
        self._append_project_exploration_reflection(context, explorer, blocked_final=blocked_final)
        call = explorer.next_tool_call(self._registered_project_read_tools())
        if not call:
            return False
        call = self._ensure_function_call_ids([call], tool_round)[0]
        fn_name = call.get("name", "")
        fn_args = call.get("args", {})
        fn_call_msg = Message(
            role="assistant",
            content=f"[调用工具: {fn_name}]",
            metadata={"function_call": call, "function_calls": [call]},
        )
        self.memory.add_message(fn_call_msg)
        context.append(fn_call_msg)

        try:
            tool_msg, plan_changed = self._complete_tool_call(call)
        except PermissionPending:
            raise
        self.memory.add_message(tool_msg)
        context.append(tool_msg)
        explorer.observe_tool(fn_name, fn_args, tool_msg.content)
        self._append_batch_context_updates(context, bool(plan_changed))
        self._append_project_exploration_reflection(context, explorer)
        return True

    def _run_project_explorer_auto_step_events(
        self,
        context: List[Message],
        tools: List[Dict[str, Any]],
        explorer: ProjectExplorer,
        user_input: str,
        tool_round: int,
        blocked_final: str = "",
    ) -> Generator[AgentEvent, None, bool]:
        self._append_project_exploration_reflection(context, explorer, blocked_final=blocked_final)
        call = explorer.next_tool_call(self._registered_project_read_tools())
        if not call:
            return False
        call = self._ensure_function_call_ids([call], tool_round)[0]
        fn_name = call.get("name", "")
        fn_call_msg = Message(
            role="assistant",
            content=f"[调用工具: {fn_name}]",
            metadata={"function_call": call, "function_calls": [call]},
        )
        self.memory.add_message(fn_call_msg)
        context.append(fn_call_msg)

        before = len(context)
        plan_changed = yield from self._process_tool_call_batch_events(
            context=context,
            tools=tools,
            function_calls=[call],
            start_index=0,
            tool_round=tool_round,
            pending_text=None,
            plan_changed=False,
        )
        if plan_changed is None:
            return True
        self._observe_explorer_from_function_calls(explorer, context, [call], start_index=before)
        self._append_batch_context_updates(context, bool(plan_changed))
        if plan_changed:
            yield self._event(PLANNING_UPDATE, self._current_planning_payload())
            yield self._event(TODO_UPDATE, self._todo_payload())
        self._append_project_exploration_reflection(context, explorer)
        return True

    @staticmethod
    def _append_project_exploration_reflection(
        context: List[Message],
        explorer: ProjectExplorer,
        blocked_final: str = "",
    ) -> None:
        context.append(Message(role="system", content=explorer.reflection_prompt(blocked_final=blocked_final)))

    def _rebuild_explorer_from_context(self, explorer: ProjectExplorer, context: List[Message]) -> None:
        for msg in context:
            if msg.role == "tool" and (msg.name or "") in PROJECT_ANALYSIS_READ_TOOLS:
                explorer.observe_tool(msg.name or "", {}, msg.content)

    @staticmethod
    def _observe_explorer_from_function_calls(
        explorer: ProjectExplorer,
        context: List[Message],
        function_calls: List[dict],
        start_index: int = 0,
    ) -> None:
        tool_messages = [msg for msg in context[start_index:] if msg.role == "tool"]
        for call, tool_msg in zip(function_calls, tool_messages):
            name = call.get("name", "")
            if name in PROJECT_ANALYSIS_READ_TOOLS:
                explorer.observe_tool(name, call.get("args", {}), tool_msg.content)

    def _force_final_answer_from_tool_context(
        self,
        context: List[Message],
        user_input: str,
        explorer: Optional[ProjectExplorer] = None,
    ) -> str:
        model_reply = self._project_analysis_model_final_answer(context, user_input)
        if model_reply and (explorer is None or explorer.final_has_evidence(model_reply)):
            return self._finalize_response(model_reply, user_input)
        reply = self._project_analysis_observation_answer(context, user_input)
        return self._finalize_response(reply, user_input)

    def _project_analysis_model_final_answer(self, context: List[Message], user_input: str) -> str:
        observations = self._project_analysis_observation_texts(context)
        if not observations:
            return ""
        observation_digest = self._project_analysis_observation_digest(observations)
        final_context = self._project_analysis_final_context(context, user_input, observation_digest)
        try:
            response = self._normalize_llm_response(self.llm.generate(messages=final_context))
            text = response.get("text")
            if not text:
                return ""
            reply = self._sanitize_plain_reply(text, user_input)
            if (
                reply
                and not self._is_invalid_plain_retry(reply)
                and not self._is_plain_fallback(reply, user_input)
                and not self._contains_pseudo_tool_text(reply)
            ):
                return reply
        except Exception as exc:
            logger.warning(f"项目分析最终总结失败，使用本地兜底: {exc}")
        return ""

    def _project_analysis_final_context(
        self,
        context: List[Message],
        user_input: str,
        observation_digest: str,
    ) -> List[Message]:
        final_context: List[Message] = []
        for msg in context:
            metadata = msg.metadata or {}
            if msg.role == "tool":
                continue
            if msg.role == "assistant" and (metadata.get("function_call") or metadata.get("function_calls")):
                continue
            if msg.role in {"system", "user"}:
                final_context.append(Message(role=msg.role, content=msg.content))

        final_context.append(Message(
            role="system",
            content=(
                "【项目分析最终总结模式】\n"
                "工具读取阶段已经结束，当前不会再提供任何工具 schema。"
                "禁止输出 [调用工具: ...]、read_file、tool_calls、DSML 或任何工具调用占位。\n"
                "请只基于下面的项目观察做深入分析，回答要比简单概括更具体，"
                "必须包含：1. 项目定位/架构判断；2. 亮点；3. 难点/风险；4. 可改进建议；5. 依据文件或模块。\n"
                "如果证据不足，请说明证据边界，但仍需基于已有观察给出结论。\n\n"
                f"【用户问题】{user_input}\n\n"
                f"【项目观察摘要】\n{observation_digest}"
            ),
        ))
        final_context.append(Message(role="user", content="请现在给出最终项目分析，不要再调用工具。"))
        return final_context

    @staticmethod
    def _project_analysis_observation_digest(observations: List[str]) -> str:
        parts = []
        for index, observation in enumerate(observations, 1):
            cleaned = str(observation or "").strip()
            if not cleaned:
                continue
            if len(cleaned) > 2500:
                cleaned = cleaned[:2500] + "\n...(单条观察已截断)"
            parts.append(f"### Observation {index}\n{cleaned}")
        digest = "\n\n".join(parts)
        if len(digest) > 16000:
            digest = digest[:16000] + "\n...(项目观察摘要已截断)"
        return digest


    def _project_analysis_observation_answer(self, context: List[Message], user_input: str) -> str:
        observations = self._project_analysis_observation_texts(context)
        joined = "\n".join(observations)
        lower_joined = joined.lower()

        observed_files = self._observed_file_hints(joined)
        capabilities = []
        capability_checks = (
            ("ReAct + Function Calling 工具循环", ("function_call", "function calling", "tool_calls", "react")),
            ("Planning / Todo / Scratchpad 状态管理", ("planning", "todo", "scratchpad", "计划")),
            ("权限分级和 Web 权限恢复", ("permission", "权限", "resume_events")),
            ("上下文压缩、短期记忆和长期记忆检索", ("context", "memory", "longmemory", "compression", "上下文", "记忆")),
            ("Web API、SSE 事件流和静态前端", ("server", "sse", "fastapi", "web/", "run_events")),
            ("项目工具层和文件/搜索/执行工具", ("read_file", "grep", "ls", "execute_code", "tools/")),
        )
        for label, needles in capability_checks:
            if any(needle in lower_joined for needle in needles):
                capabilities.append(label)

        if not capabilities:
            capabilities = [
                "分层的 Agent Runtime 和工具系统",
                "面向代码项目的只读分析、编辑和验证能力",
            ]

        file_line = "、".join(observed_files[:6]) if observed_files else "本轮已读取的项目结构和关键文件片段"
        capability_line = "；".join(capabilities[:4])

        return (
            "基于本轮已经获取到的项目观察，我先停止继续读取文件，直接给出当前可判断的结论。\n\n"
            "**亮点**\n"
            f"- 项目能力边界比较完整：{capability_line}，不是单纯聊天或纯 RAG。\n"
            "- 结构上已经把 runtime、context、memory、planning、permission、tools、server/web 等职责拆开，后续定位问题和扩展工具比较清晰。\n"
            "- 对真实 code agent 的关键问题已有覆盖：工具协议、权限确认、上下文压缩、长期记忆、任务计划、SSE 前端事件等都有独立模块和测试。\n\n"
            "**难点**\n"
            "- 最大难点是工具调用闭环的稳定性：模型可能反复请求 read_file、输出伪工具文本，运行时必须用硬规则兜住最终回答。\n"
            "- 上下文管理复杂：Chat History、Tool Results、Scratchpad、Memory、Planning 都会进入 prompt，任何一层污染都可能导致下一轮继续误调工具。\n"
            "- Web 侧要把 tool_call/tool_result/thinking/final 分开展示，否则内部执行状态很容易和最终回答混在一起。\n"
            "- 项目越接近真实 agent，难点越从“能调用工具”转向“何时停止、如何恢复、如何保证协议和用户可见状态一致”。\n\n"
            f"**依据**：{file_line}。如果要更精确的代码级评价，可以指定某个目录，我会只围绕该目录分析。"
        )

    @staticmethod
    def _project_analysis_observation_texts(context: List[Message]) -> List[str]:
        texts = []
        for msg in context:
            if msg.role == "tool" and (msg.name or "") in PROJECT_ANALYSIS_READ_TOOLS:
                content = str(msg.content or "").strip()
                if content:
                    texts.append(content[:4000])
        return texts

    @staticmethod
    def _observed_file_hints(observation_text: str) -> List[str]:
        hints: List[str] = []
        patterns = (
            r"文件:\s*([^\n]+)",
            r"目录:\s*([^\n]+)",
            r"(^|\n)([A-Za-z0-9_./-]+\.(?:py|md|txt|js|css|json|html))",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, observation_text):
                value = match.group(match.lastindex or 1).strip()
                value = value.split("(", 1)[0].strip()
                if value and value not in hints:
                    hints.append(value)
        return hints


    @staticmethod
    def _normalize_llm_response(response: Any) -> Dict[str, Any]:
        if isinstance(response, dict):
            return response
        if isinstance(response, str):
            return {"text": response, "function_call": None, "function_calls": []}
        return {"text": None, "function_call": None, "function_calls": []}

    @staticmethod
    def _extract_function_calls(response: Dict[str, Any]) -> List[dict]:
        calls = response.get("function_calls")
        if isinstance(calls, list) and calls:
            return [call for call in calls if isinstance(call, dict)]

        call = response.get("function_call")
        if isinstance(call, dict):
            return [call]

        return []

    @staticmethod
    def _ensure_function_call_ids(function_calls: List[dict], round_index: int) -> List[dict]:
        normalized = []
        for index, call in enumerate(function_calls):
            copied = dict(call)
            if not copied.get("id"):
                name = copied.get("name", "tool")
                copied["id"] = f"call_{round_index}_{index}_{abs(hash(name))}"
            normalized.append(copied)
        return normalized

    @staticmethod
    def _truncate_tool_result(tool_result: str, limit: int = 8000) -> str:
        if not isinstance(tool_result, str):
            tool_result = str(tool_result)
        if len(tool_result) <= limit:
            return tool_result
        trunc_note = f"\n\n...(结果过长，已截断。共 {len(tool_result)} 字符，显示前 {limit} 字符)"
        return tool_result[:limit] + trunc_note

    def _run_with_callback_tool(self) -> str:
        raw_history = self.memory.get_recent_messages()
        history = self._plain_chat_history(raw_history)
        user_text = self._last_user_content(history)
        rag_results = self._get_rag()
        planning_context = self._get_planning_context()

        tool_results = None
        if self.tool_fn:
            try:
                tool_result = self.tool_fn(user_text)
                tool_results = [tool_result] if tool_result else None
            except Exception as e:
                logger.warning(f"Tool 执行失败: {e}")
                tool_results = [f"[Tool 执行出错: {e}]"]

        long_memory_context = self._get_long_memory_context()
        todo_context = self._get_todo_context()
        scratchpad_context = self._get_scratchpad_context()

        context = build_context(
            system_prompt=self.system_prompt,
            messages=history,
            rag_results=rag_results,
            tool_results=tool_results,
            long_memory_context=long_memory_context,
            project_context=self.project_context,
            todo_context=todo_context,
            planning_context=planning_context,
            scratchpad_context=scratchpad_context,
        )
        context.append(self._plain_answer_mode_message())

        try:
            response = self.llm.generate(messages=context)
            raw_reply = response if isinstance(response, str) else response.get("text", str(response))
            reply = self._sanitize_plain_reply(raw_reply, user_text)
            if self._is_generic_plain_fallback(reply, user_text) and self._contains_pseudo_tool_text(raw_reply):
                retry = self._retry_plain_answer(context, user_text)
                if (
                    retry
                    and not self._is_plain_fallback(retry, user_text)
                    and not self._is_invalid_plain_retry(retry)
                ):
                    reply = retry
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            reply = f"[抱歉，我遇到了一个错误: {str(e)[:200]}]"

        return self._finalize_response(reply)

    def _plain_chat_history(self, messages: List[Message]) -> List[Message]:
        """Return chat history suitable for a no-tools answer turn.

        Previous native tool runs store assistant messages with function_calls
        and tool result messages so the next tool round can satisfy the API
        protocol. For a plain follow-up, sending those protocol messages back
        tends to make the model emit read_file/tool_calls text again. Keep the
        user-visible conversation and drop the transport-level tool protocol.
        """
        plain: List[Message] = []
        for msg in messages:
            if msg.role == "tool":
                continue
            metadata = msg.metadata or {}
            if msg.role == "assistant" and (
                metadata.get("function_calls") or metadata.get("function_call")
            ):
                continue
            if msg.role not in {"user", "assistant", "system"}:
                continue
            plain.append(Message(
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp,
                metadata={},
                name=msg.name,
            ))
        return plain

    @staticmethod
    def _last_user_content(messages: List[Message]) -> str:
        for msg in reversed(messages):
            if msg.role == "user":
                return msg.content
        return ""

    @staticmethod
    def _plain_answer_mode_message() -> Message:
        return Message(
            role="system",
            content=(
                "【本轮执行模式】普通回答模式。当前没有向模型暴露任何工具 schema，"
                "禁止输出 DSML、tool_calls、invoke、read_file 等任何伪工具调用标记；"
                "不要说‘本轮没有实际调用工具’这类占位话术。"
                "如果用户是在追问刚才的项目分析，请基于已有对话结论直接回答观点、理由和取舍。"
            ),
        )

    @staticmethod
    def _native_tool_retry_message() -> Message:
        return Message(
            role="system",
            content=(
                "上一条输出只是工具调用草稿，但没有产生合法 function_call。"
                "如果需要查看、搜索、修改或运行项目，请立即使用已暴露的工具 schema；"
                "如果不需要工具，请直接给最终答案。禁止输出 DSML、tool_calls、invoke、read_file 等伪工具文本。"
            ),
        )

    def _is_unusable_tool_draft(self, text: str) -> bool:
        return self._contains_pseudo_tool_text(text) or self._looks_like_tool_preamble(text)

    def _retry_plain_answer(self, context: List[Message], user_input: str) -> str:
        retry_context = list(context)
        retry_context.append(Message(
            role="system",
            content=(
                "上一条输出仍然像工具调用草稿。请忽略任何工具调用意图，"
                "只用自然语言直接回答用户问题；不要提 read_file、tool_calls 或权限。"
            ),
        ))
        retry_context.append(Message(
            role="user",
            content=f"请直接回答这个问题：{user_input}",
        ))
        try:
            response = self.llm.generate(messages=retry_context)
            raw_reply = response if isinstance(response, str) else response.get("text", str(response))
            return self._sanitize_plain_reply(raw_reply, user_input)
        except Exception as e:
            logger.warning(f"普通回答重试失败: {e}")
            return ""

    @staticmethod
    def _contains_pseudo_tool_text(reply: str) -> bool:
        text = str(reply or "")
        return any(marker in text for marker in PSEUDO_TOOL_MARKERS)

    def _is_plain_fallback(self, reply: str, user_input: str) -> bool:
        return str(reply or "").strip() == self._fallback_plain_answer(user_input).strip()

    def _is_generic_plain_fallback(self, reply: str, user_input: str) -> bool:
        fallback = self._fallback_plain_answer(user_input).strip()
        return (
            str(reply or "").strip() == fallback
            and fallback.startswith("这个问题属于普通问答")
        )

    @staticmethod
    def _is_invalid_plain_retry(reply: str) -> bool:
        text = str(reply or "").strip().lower()
        if not text:
            return True
        invalid_markers = (
            "[no more responses]",
            "[抱歉，我遇到了一个错误",
            "agent 未返回有效响应",
        )
        return any(marker in text for marker in invalid_markers)

    def stream_run(self, user_input: str) -> Generator[str, None, None]:
        """Compatibility streaming facade built on the canonical event loop.

        The old implementation had a separate prompt/tool path. Keeping this
        method as a thin wrapper ensures CLI/legacy callers get the same
        Planning, permission, memory, and pseudo-tool cleanup behavior as Web.
        """
        for event in self.run_events(user_input):
            if event.type == FINAL:
                content = event.data.get("content", "")
                if content:
                    yield content
            elif event.type == ERROR:
                message = event.data.get("message", "未知错误")
                yield f"[错误] {message}"
            elif event.type == PERMISSION_REQUEST:
                tool_name = event.data.get("tool_name", "工具调用")
                yield f"[等待权限确认] {tool_name} 需要用户批准后才能继续。"

    def _get_rag(self) -> Optional[List[str]]:
        if not self.rag_fn:
            return None
        try:
            last_user = self.memory.get_last_user_message()
            query = last_user.content if last_user else ""
            return self.rag_fn(query)
        except Exception as e:
            logger.warning(f"RAG 检索失败: {e}")
            return None

    def _get_long_memory_context(self) -> Optional[str]:
        if self.long_memory:
            last_user = self.memory.get_last_user_message()
            query_parts = []
            if last_user and last_user.content:
                query_parts.append(last_user.content)
            if self.planning_manager is not None and self.planning_manager.state.objective:
                query_parts.append(self.planning_manager.state.objective)
                query_parts.extend(self.planning_manager.state.steps[-3:])
            query = "\n".join(query_parts) if query_parts else None
            try:
                return self.long_memory.get_context(query=query)
            except TypeError:
                try:
                    return self.long_memory.get_context()
                except Exception as e:
                    logger.warning(f"长期记忆上下文获取失败: {e}")
                    return None
            except Exception as e:
                logger.warning(f"长期记忆上下文获取失败: {e}")
                return None
        return None

    def _get_todo_context(self) -> Optional[str]:
        if self.todo_list and hasattr(self.todo_list, 'format_for_prompt'):
            try:
                return self.todo_list.format_for_prompt()
            except Exception as e:
                logger.warning(f"TodoList 状态获取失败: {e}")
        return None

    def _build_context(
        self,
        rag_results: Optional[List[str]] = None,
        long_memory_context: Optional[str] = None,
        todo_context: Optional[str] = None,
        planning_context: Optional[str] = None,
        scratchpad_context: Optional[str] = None,
    ) -> List[Message]:
        history = self._tool_protocol_safe_history(self.memory.get_recent_messages())
        return build_context(
            system_prompt=self.system_prompt,
            messages=history,
            rag_results=rag_results,
            long_memory_context=long_memory_context,
            project_context=self.project_context,
            todo_context=todo_context,
            planning_context=planning_context,
            scratchpad_context=scratchpad_context,
        )

    def _append_tool_workflow_context(self, context: List[Message], user_input: str) -> None:
        message = self._tool_workflow_message(user_input)
        if message is not None:
            context.append(message)

    def _tool_workflow_message(self, user_input: str) -> Optional[Message]:
        lower = str(user_input or "").strip().lower()
        if self._is_project_analysis_request(lower):
            return Message(
                role="system",
                content=(
                    "【项目分析工具工作流】\n"
                    "本轮是只读项目/目录分析。先用 ls 或 grep 获取结构线索，再只读取必要的关键文件。"
                    "read_file 无行号时只会返回文件开头摘要；如需更多内容，必须指定 start_line/end_line。\n"
                    "Runtime 会跟踪 checklist 覆盖率和信息增益；核心模块未覆盖时，应继续调用工具探索。"
                    "不要尝试写文件、执行代码或联网搜索。"
                ),
            )
        return None

    def _tool_protocol_safe_history(self, messages: List[Message]) -> List[Message]:
        """Drop truncated function-calling fragments before native tool calls.

        ShortMemory trims from the front by message count/token count. That can
        leave a leading tool message or an assistant tool_calls message without
        every matching tool response. OpenAI-compatible APIs reject that shape,
        so native tool contexts must only keep complete protocol batches.
        """
        safe: List[Message] = []
        index = 0
        while index < len(messages):
            msg = messages[index]

            if msg.role == "tool":
                index += 1
                continue

            function_calls = self._message_function_calls(msg)
            if msg.role == "assistant" and function_calls:
                expected_ids = [
                    str(call.get("id"))
                    for call in function_calls
                    if call.get("id")
                ]
                valid = (
                    len(expected_ids) == len(function_calls)
                    and len(set(expected_ids)) == len(expected_ids)
                )
                batch_tools: List[Message] = []
                seen_ids: set[str] = set()
                cursor = index + 1

                while cursor < len(messages) and messages[cursor].role == "tool":
                    tool_msg = messages[cursor]
                    response = (tool_msg.metadata or {}).get("function_response", {})
                    tool_call_id = response.get("id")
                    if not tool_call_id:
                        valid = False
                    else:
                        tool_call_id = str(tool_call_id)
                        if tool_call_id not in expected_ids or tool_call_id in seen_ids:
                            valid = False
                        else:
                            seen_ids.add(tool_call_id)
                            batch_tools.append(tool_msg)
                    cursor += 1

                if valid and all(call_id in seen_ids for call_id in expected_ids):
                    safe.append(msg)
                    safe.extend(batch_tools)

                index = cursor
                continue

            safe.append(msg)
            index += 1

        return safe

    @staticmethod
    def _message_function_calls(msg: Message) -> List[dict]:
        metadata = msg.metadata or {}
        calls = metadata.get("function_calls")
        if isinstance(calls, list) and calls:
            return [call for call in calls if isinstance(call, dict)]
        call = metadata.get("function_call")
        if isinstance(call, dict):
            return [call]
        return []

    def _start_scratchpad(self, user_input: str) -> None:
        if self.scratchpad is not None:
            self.scratchpad.set_objective(user_input)

    def _get_scratchpad_context(self) -> Optional[str]:
        if self.scratchpad is None:
            return None
        try:
            return self.scratchpad.format_for_prompt()
        except Exception as e:
            logger.warning(f"Scratchpad 上下文获取失败: {e}")
            return None

    def _observe_scratchpad(self, tool_name: str, args: dict, observation: str) -> None:
        if self.scratchpad is None:
            return
        try:
            self.scratchpad.observe_tool_result(
                tool_name,
                args if isinstance(args, dict) else {},
                observation,
            )
        except Exception as e:
            logger.warning(f"Scratchpad 更新失败: {e}")

    @staticmethod
    def _scratchpad_context_message(scratchpad_context: str) -> Message:
        return Message(
            role="system",
            content=(
                "【当前任务草稿区已更新】\n"
                f"{scratchpad_context}\n"
                "请基于这些显式工作笔记继续推进。"
            ),
            metadata={"scratchpad_context": True},
        )

    def _maybe_start_plan(self, user_input: str):
        if self.planning_manager is None:
            return None
        try:
            update = self.planning_manager.start_or_update_plan(user_input, getattr(self, "llm", None))
            if update.changed:
                logger.info(f"Planning 已启动: {update.reason}")
                if self.scratchpad is not None:
                    self.scratchpad.merge_next_steps(list(update.plan.steps))
                self._save_todo_if_available()
            return update
        except Exception as e:
            logger.warning(f"Planning 启动失败: {e}")
            return None

    def _update_plan_from_observation(self, tool_name: str, args: dict, observation: str):
        if self.planning_manager is None:
            return None
        try:
            update = self.planning_manager.observe_tool_result(tool_name, args, observation)
            if update.changed:
                logger.info(f"Planning 已更新: {update.reason}")
                if self.scratchpad is not None:
                    self.scratchpad.merge_next_steps(list(update.plan.steps[-3:]))
                self._save_todo_if_available()
            return update
        except Exception as e:
            logger.warning(f"Planning 更新失败: {e}")
            return None

    def _get_planning_context(self) -> Optional[str]:
        if self.planning_manager is None:
            return None
        try:
            return self.planning_manager.format_for_prompt()
        except Exception as e:
            logger.warning(f"Planning 上下文获取失败: {e}")
            return None

    def _complete_tool_call(
        self,
        function_call: dict,
        permission_override: Optional[bool] = None,
    ) -> tuple[Message, bool]:
        fn_name = function_call.get("name", "")
        fn_args = function_call.get("args", {})
        tool_result = self._execute_tool(fn_name, fn_args, permission_override)
        tool_content = self._truncate_tool_result(tool_result)

        tool_msg = Message(
            role="tool",
            content=tool_content,
            name=fn_name,
            metadata={
                "function_response": {
                    "id": function_call.get("id"),
                    "name": fn_name,
                    "response": {"result": tool_content},
                }
            },
        )

        self._observe_scratchpad(fn_name, fn_args, tool_content)

        planning_update = None
        if self.planning_manager is not None:
            planning_update = self._update_plan_from_observation(
                fn_name,
                fn_args if isinstance(fn_args, dict) else {},
                tool_content,
            )

        return tool_msg, bool(planning_update and planning_update.changed)

    def _planning_context_message(self, planning_context: str) -> Message:
        return Message(
            role="system",
            content=(
                "【当前执行计划已更新】\n"
                f"{planning_context}\n"
                "请依据更新后的计划继续执行。"
            ),
        )

    def _event(self, event_type: str, data: Optional[dict] = None) -> AgentEvent:
        return make_event(event_type, data, self.session_id)

    @staticmethod
    def _tool_call_payload(function_call: dict) -> dict:
        return {
            "id": function_call.get("id"),
            "name": function_call.get("name", ""),
            "args": function_call.get("args", {}),
        }

    @staticmethod
    def _tool_result_payload(tool_msg: Message) -> dict:
        response = tool_msg.metadata.get("function_response", {})
        return {
            "id": response.get("id"),
            "name": response.get("name") or tool_msg.name,
            "content": tool_msg.content,
        }

    @staticmethod
    def _planning_payload(update: Any) -> dict:
        plan = update.plan
        return {
            "changed": update.changed,
            "reason": update.reason,
            "mode": plan.mode,
            "objective": plan.objective,
            "steps": list(plan.steps),
            "revision": plan.revision,
            "last_observation": plan.last_observation,
        }

    def _current_planning_payload(self) -> dict:
        if self.planning_manager is None:
            return {"changed": False, "mode": "react", "steps": []}
        state = self.planning_manager.state
        return {
            "changed": True,
            "reason": "Planning 状态已更新",
            "mode": state.mode,
            "objective": state.objective,
            "steps": list(state.steps),
            "revision": state.revision,
            "last_observation": state.last_observation,
        }

    def _todo_payload(self) -> dict:
        if self.todo_list is None:
            return {"enabled": False, "text": None, "progress": 0.0}
        return {
            "enabled": True,
            "text": self._get_todo_context(),
            "progress": self.todo_list.progress() if hasattr(self.todo_list, "progress") else 0.0,
        }

    def _save_todo_if_available(self) -> None:
        store = getattr(self, "todo_store", None)
        if store is not None and self.todo_list is not None:
            try:
                store.save(self.todo_list)
            except Exception as e:
                logger.warning(f"TodoList 持久化失败: {e}")

    def _execute_tool(
        self,
        name: str,
        args: dict,
        permission_override: Optional[bool] = None,
    ) -> str:
        if not isinstance(args, dict):
            return f"[错误] 工具参数必须是 JSON 对象，实际收到: {type(args).__name__}"

        if permission_override is False:
            return f"[权限拒绝] Tool '{name}' 需要用户确认，但未获批准。"

        if self.permission_fn is not None and permission_override is None:
            try:
                approved = self.permission_fn(name, args)
                if not approved:
                    return f"[权限拒绝] Tool '{name}' 需要用户确认，但未获批准。"
            except PermissionPending:
                raise
            except Exception as e:
                logger.warning(f"权限检查异常: {e}")

        try:
            result = self.tool_registry.execute(name, **args)
            logger.debug(f"Tool '{name}' 执行成功: {len(result)} 字符")
            return result
        except KeyError:
            return f"[错误] 未知工具: '{name}'。可用工具: {self.tool_registry.list_names()}"
        except ValueError as e:
            return f"[错误] 工具参数错误: {e}"
        except Exception as e:
            logger.error(f"Tool '{name}' 执行异常: {e}")
            return f"[错误] 工具执行异常: {str(e)[:300]}"

    def _sanitize_plain_reply(self, reply: str, user_input: str = "") -> str:
        text = str(reply or "").strip()
        if not text:
            return text
        if not any(marker in text for marker in PSEUDO_TOOL_MARKERS):
            if self._looks_like_tool_preamble(text):
                return self._fallback_plain_answer(user_input)
            return text

        cleaned = re.sub(
            r"<｜｜DSML｜｜tool_calls>.*?</｜｜DSML｜｜tool_calls>",
            "",
            text,
            flags=re.DOTALL,
        )
        cleaned = re.sub(r"<\|[^>]*tool_calls[^>]*>.*?</\|[^>]*tool_calls[^>]*>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<tool_calls?>.*?</tool_calls?>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<｜｜DSML｜｜invoke.*?</｜｜DSML｜｜invoke>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<｜｜DSML｜｜parameter.*?</｜｜DSML｜｜parameter>", "", cleaned, flags=re.DOTALL)
        cleaned = "\n".join(
            line.strip()
            for line in cleaned.splitlines()
            if line.strip() and not self._looks_like_tool_preamble(line)
        ).strip()

        if cleaned and not self._looks_like_tool_preamble(cleaned):
            return cleaned

        return self._fallback_plain_answer(user_input)

    @staticmethod
    def _looks_like_tool_preamble(text: str) -> bool:
        compact = " ".join(str(text or "").strip().split())
        if not compact:
            return False
        lowered = compact.lower()

        protocol_prefixes = ("read_file", "tool_calls", "tool_call", "invoke", "dsml", "<｜｜", "<|", "[调用工具", "调用工具:")
        if lowered.startswith(protocol_prefixes):
            return True

        answer_markers = ("难点", "亮点", "结论", "总结", "建议", "原因", "包括", "如下", "主要")
        if len(compact) > 120 and any(marker in compact for marker in answer_markers):
            return False

        draft_patterns = (
            r"^(让我|我来|我现在来|我先|先|首先|接下来|下面).{0,40}(用实际代码|看看|看一下|查看|检查|读取|搜索|搜一下|查找|分析|打开)",
            r"^(需要|应该|最好).{0,30}(查看|检查|读取|搜索|查找)",
        )
        if any(re.search(pattern, compact, flags=re.IGNORECASE) for pattern in draft_patterns):
            return True

        short_draft_phrases = (
            "用实际代码",
            "看看相关",
            "查看相关",
            "先看看",
            "先看一下",
            "让我先看",
            "让我先检查",
            "我现在来检查",
            "让我检查",
            "先检查",
        )
        if len(compact) <= 140 and any(phrase in lowered for phrase in short_draft_phrases):
            return True

        english_draft_patterns = (
            r"^(let me|i'?ll|i am going to|i'm going to|i will|i need to|i want to|trying to).{0,80}(search|fetch|look up|check|get|try|retrieve|find|browse)",
            r"^(let me try|i will try|i'll try).{0,80}(search|fetch|look up|check|get|retrieve)",
        )
        return any(re.search(pattern, compact, flags=re.IGNORECASE) for pattern in english_draft_patterns)

    @staticmethod
    def _fallback_plain_answer(user_input: str) -> str:
        text = str(user_input or "")
        lower = text.lower()
        if "复杂任务" in text or "planning" in lower or "规划" in text:
            return (
                "对于复杂任务，我会先进入 Planning 模式，把目标拆成可执行步骤并同步到 Todo；"
                "随后按步骤进入 ReAct 循环，必要时才调用工具查看、修改或运行项目内容。"
                "每次工具结果会更新 Scratchpad、Planning 和 Todo；如果观察到错误、失败或权限拒绝，"
                "会补充验证步骤或重新规划。最后会运行必要检查并给出结果总结。"
                "像这类询问机制的问题属于普通问答，不应该真实调用工具。"
            )
        if "上下文" in text or "context" in lower:
            return (
                "上下文由 system prompt、用户消息、历史对话、工具结果、Scratchpad、长期记忆、"
                "项目上下文、Planning 和 Todo 状态共同组成。运行时会按当前问题检索相关记忆，"
                "必要时压缩旧历史，并把显式中间状态写入 Scratchpad；普通问答不会触发工具调用。"
            )
        if any(keyword in text for keyword in ("记忆", "memory", "历史记录", "历史", "长期记忆", "会话", "session")):
            storage_answer = (
                "记忆和历史现在分两层保存：Web/API 会话的用户可见聊天记录保存在 "
                "`.agent_sessions/<session_id>/history.json`，左侧历史会话列表和重新打开页面主要从这里恢复；"
                "长期记忆 `LongMemory` 保存在 `.agent_sessions/<session_id>/memory_long.json`，"
                "用于结构化保存摘要、事实、偏好、测试结论等可检索信息；Todo 状态保存在 "
                "`.agent_sessions/<session_id>/todo.json`。打开旧会话时，`AgentSessionManager` 会扫描 "
                "`.agent_sessions`，读取对应的 `history.json` 恢复可见对话，并把 user/assistant 历史放回短期记忆以支持继续追问。"
            )
            if any(keyword in text for keyword in ("哪里", "哪儿", "在哪", "从哪里", "存储", "保存", "恢复")):
                return storage_answer
            return (
                "记忆模块目前已经不是只等 8 轮后才摘要保存：用户可见会话历史会逐轮持久化，"
                "长期记忆会结构化保存并按当前问题检索相关内容。"
                f"{storage_answer}"
            )
        return "这个问题属于普通问答，本轮没有实际调用工具。我会直接根据已有上下文用自然语言回答。"

    def _finalize_response(self, reply: str, user_input: str = "") -> str:
        if not user_input:
            last_user = self.memory.get_last_user_message()
            user_input = last_user.content if last_user else ""
        reply = self._sanitize_plain_reply(reply, user_input)
        assistant_message = Message(role="assistant", content=reply)
        self.memory.add_message(assistant_message)
        self._maybe_auto_summarize()
        return reply

    def _maybe_auto_summarize(self):
        self._conversation_turns += 1
        if self.long_memory and self._conversation_turns >= self.auto_summarize_threshold:
            self._auto_summarize()
            self._conversation_turns = 0

    def _auto_summarize(self):
        if not self.long_memory or not self.memory.messages:
            return
        recent = self.memory.get_recent_messages(6)
        if len(recent) < 2:
            return
        try:
            self.long_memory.summarize_and_store(recent)
            logger.info(f"自动总结完成: {len(recent)} 条消息 -> 长期记忆")
        except Exception as e:
            logger.warning(f"自动总结失败: {e}")

    def _check_and_compress(self):
        if self.context_compressor is None:
            return
        try:
            if hasattr(self.context_compressor, 'should_compress') and \
               self.context_compressor.should_compress(self.memory.messages):
                compression_result = self.context_compressor.compress(
                    self.memory.messages, self.llm
                )
                if hasattr(compression_result, "messages"):
                    self.memory.replace_messages(compression_result.messages)
                    if self.long_memory and getattr(compression_result, "summary", None):
                        self.long_memory.add_summary(compression_result.summary)
                    logger.info(
                        "上下文压缩完成: "
                        f"{compression_result.compressed_count} 条消息, "
                        f"~{compression_result.before_tokens} -> ~{compression_result.after_tokens} tokens"
                    )
                elif isinstance(compression_result, list):
                    self.memory.replace_messages(compression_result)
                    logger.info(f"上下文压缩完成: {len(compression_result)} 条消息")
                else:
                    logger.info("上下文压缩器返回空结果，跳过替换")
        except Exception as e:
            logger.warning(f"上下文压缩失败: {e}")

    def add_fact(self, fact: str):
        if self.long_memory:
            self.long_memory.add_fact(fact)

    def get_conversation_summary(self) -> str:
        parts = [
            f"当前对话轮数: {self._conversation_turns}",
            f"短期记忆消息数: {len(self.memory.messages)}",
            f"短期记忆 token 数: ~{self.memory.total_tokens()}",
        ]
        if self.long_memory:
            parts.append(f"长期记忆事实数: {len(self.long_memory.facts)}")
            parts.append(f"长期记忆摘要数: {len(self.long_memory.summaries)}")
            if hasattr(self.long_memory, "items"):
                parts.append(f"长期记忆条目数: {len(self.long_memory.items)}")
        if self.tool_registry:
            parts.append(f"可用工具: {self.tool_registry.list_names()}")
        if self.project_context:
            parts.append(f"AGENT.md: 已加载 ({len(self.project_context)} 字符)")
        if self.todo_list:
            parts.append(f"TodoList: 已配置")
        if self.scratchpad and not self.scratchpad.is_empty():
            parts.append("Scratchpad: 已记录当前任务状态")
        return "\n".join(parts)

    def reset_memory(self):
        self.memory.clear()
        if self.scratchpad is not None:
            self.scratchpad.clear()
        self._conversation_turns = 0

    def reset_long_memory(self):
        if self.long_memory:
            self.long_memory.clear()

    def reset_all(self):
        self.reset_memory()
        self.reset_long_memory()
