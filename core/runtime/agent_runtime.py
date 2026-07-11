import logging
import re
from typing import List, Optional, Generator, Callable, Any, Dict
from uuid import uuid4
from core.message import Message
from core.memory import ShortMemory, LongMemory
from core.context import Scratchpad, build_context
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
from core.web_permission import PermissionPending
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 12  # 复杂任务（如分析多文件目录）需要更多工具轮次

TOOL_ACTION_KEYWORDS = (
    "修复", "修改", "实现", "新增", "添加", "删除", "重构", "接入", "集成",
    "测试", "运行", "执行", "检查", "调试", "定位", "复现", "报错", "失败",
    "读取", "打开", "搜索", "查找", "grep", "diff",
    "补丁", "提交", "联网", "fix", "implement", "refactor", "debug", "test",
    "run", "execute", "traceback", "exception", "failed", "commit",
)

TOOL_RESOURCE_KEYWORDS = (
    "项目", "仓库", "目录", "文件", "源码", "函数", "类", "模块", "接口", "依赖",
    "配置", "日志", "页面", "前端", "后端", "服务", "路由", "数据库", "api",
    "网页", "网址", "url", "网站", "错误", "bug",
)

TOOL_INTENT_KEYWORDS = TOOL_ACTION_KEYWORDS + TOOL_RESOURCE_KEYWORDS

DIRECT_ANSWER_HINTS = (
    "你觉得", "是什么", "什么是", "为什么", "如何", "怎么", "怎样", "解释",
    "介绍", "说明", "概念", "原理", "关键", "原则", "建议", "区别",
    "优缺点", "有办法", "可以", "能否", "是否", "吗",
)

PSEUDO_TOOL_MARKERS = (
    "DSML",
    "tool_calls",
    "tool_call",
    "<｜｜",
    "<|",
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
            return self._run_with_native_tools()

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
                yield from self._run_native_tools_events()
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

        has_direct_answer_hint = any(hint in lower for hint in DIRECT_ANSWER_HINTS)
        has_action_intent = any(keyword in lower for keyword in TOOL_ACTION_KEYWORDS)
        has_resource_hint = any(keyword in lower for keyword in TOOL_RESOURCE_KEYWORDS)

        if has_direct_answer_hint and not has_action_intent:
            return False

        if has_action_intent:
            return True

        if any(marker in lower for marker in CONTINUATION_TOOL_HINTS):
            state = getattr(self.planning_manager, "state", None)
            if state is not None and getattr(state, "mode", "react") == "planning":
                return True

        return has_resource_hint and not has_direct_answer_hint

    def resume_events(self, approved: bool) -> Generator[AgentEvent, None, None]:
        """Resume a paused Web/API run after a permission decision."""
        if self._pending_tool_call is None:
            yield self._event(ERROR, {"message": "没有等待恢复的工具调用"})
            return

        context = self._pending_context
        tools = self._pending_tools
        function_call = self._pending_tool_call
        tool_round = self._pending_tool_round
        pending_text = self._pending_text

        self._pending_context = None
        self._pending_tools = None
        self._pending_tool_call = None
        self._pending_tool_round = 0
        self._pending_text = None

        if context is None or tools is None:
            yield self._event(ERROR, {"message": "等待恢复的上下文已丢失"})
            return

        tool_msg, plan_changed = self._complete_tool_call(function_call, bool(approved))
        self.memory.add_message(tool_msg)
        context.append(tool_msg)
        if plan_changed:
            refreshed_plan = self._get_planning_context()
            if refreshed_plan:
                context.append(self._planning_context_message(refreshed_plan))
            yield self._event(PLANNING_UPDATE, self._current_planning_payload())
            yield self._event(TODO_UPDATE, self._todo_payload())

        yield self._event(TOOL_RESULT, self._tool_result_payload(tool_msg))
        yield from self._continue_native_tools_events(context, tools, tool_round, pending_text)

    def _run_with_native_tools(self) -> str:
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

        tools = self.tool_registry.get_function_declarations()
        tool_round = 0
        pending_text = None  # 缓存的文本回复（用于 text + function_call 共存场景）

        while tool_round < MAX_TOOL_ROUNDS:
            tool_round += 1

            response = self._normalize_llm_response(
                self.llm.generate(messages=context, tools=tools)
            )
            text = response.get("text")
            function_calls = self._ensure_function_call_ids(
                self._extract_function_calls(response),
                tool_round,
            )

            #纯文本（无 function_call）→ 立即返回
            if text and not function_calls:
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

                    logger.info(f"LLM 请求调用 Tool ({tool_round}/{MAX_TOOL_ROUNDS}): "
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
                continue

            # 既没有 text 也没有 function_call — 异常
            if pending_text:
                return self._finalize_response(pending_text)
            return "[Agent 未返回有效响应]"

        # 循环耗尽 — 如果有缓存文本，返回它；否则报错
        if pending_text:
            logger.info(f"工具轮次用尽（{MAX_TOOL_ROUNDS}），返回缓存的文本")
            return self._finalize_response(pending_text)

        return f"[Agent 调用工具次数过多（{MAX_TOOL_ROUNDS}次），已中止]"

    def _run_native_tools_events(self) -> Generator[AgentEvent, None, None]:
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
        tools = self.tool_registry.get_function_declarations()
        yield from self._continue_native_tools_events(context, tools, 0, None)

    def _continue_native_tools_events(
        self,
        context: List[Message],
        tools: List[Dict[str, Any]],
        tool_round: int,
        pending_text: Optional[str],
    ) -> Generator[AgentEvent, None, None]:
        while tool_round < MAX_TOOL_ROUNDS:
            tool_round += 1

            response = self._normalize_llm_response(
                self.llm.generate(messages=context, tools=tools)
            )
            text = response.get("text")
            function_calls = self._ensure_function_call_ids(
                self._extract_function_calls(response),
                tool_round,
            )

            if text:
                yield self._event(ASSISTANT_TEXT, {"content": text})

            if text and not function_calls:
                reply = self._finalize_response(text)
                yield self._event(FINAL, {"content": reply})
                yield self._event(DONE, {})
                return

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

                plan_changed = False

                for function_call in function_calls:
                    yield self._event(TOOL_CALL, self._tool_call_payload(function_call))
                    try:
                        tool_msg, changed = self._complete_tool_call(function_call)
                    except PermissionPending as pending:
                        self._pending_tool_call = function_call
                        self._pending_context = context
                        self._pending_tools = tools
                        self._pending_tool_round = tool_round
                        self._pending_text = pending_text
                        yield self._event(PERMISSION_REQUEST, pending.request.to_dict())
                        return

                    self.memory.add_message(tool_msg)
                    context.append(tool_msg)
                    plan_changed = changed or plan_changed
                    yield self._event(TOOL_RESULT, self._tool_result_payload(tool_msg))

                refreshed_scratchpad = self._get_scratchpad_context()
                if refreshed_scratchpad:
                    context.append(self._scratchpad_context_message(refreshed_scratchpad))

                if plan_changed:
                    refreshed_plan = self._get_planning_context()
                    if refreshed_plan:
                        context.append(self._planning_context_message(refreshed_plan))
                    yield self._event(PLANNING_UPDATE, self._current_planning_payload())
                    yield self._event(TODO_UPDATE, self._todo_payload())
                continue

            if pending_text:
                reply = self._finalize_response(pending_text)
                yield self._event(FINAL, {"content": reply})
                yield self._event(DONE, {})
                return

            yield self._event(ERROR, {"message": "Agent 未返回有效响应"})
            return

        if pending_text:
            reply = self._finalize_response(pending_text)
            yield self._event(FINAL, {"content": reply})
            yield self._event(DONE, {})
            return

        yield self._event(ERROR, {
            "message": f"Agent 调用工具次数过多（{MAX_TOOL_ROUNDS}次），已中止",
        })

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
        history = self.memory.get_recent_messages()
        rag_results = self._get_rag()
        planning_context = self._get_planning_context()

        tool_results = None
        if self.tool_fn:
            try:
                tool_result = self.tool_fn(history[-1].content if history else "")
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
        context.append(Message(
            role="system",
            content=(
                "【本轮执行模式】普通回答模式。当前没有向模型暴露任何工具 schema，"
                "禁止输出 DSML、tool_calls、invoke、read_file 等任何伪工具调用标记；"
                "如果用户是在询问机制或概念，请直接用自然语言回答。"
            ),
        ))

        try:
            response = self.llm.generate(messages=context)
            reply = response if isinstance(response, str) else response.get("text", str(response))
            reply = self._sanitize_plain_reply(reply, history[-1].content if history else "")
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            reply = f"[抱歉，我遇到了一个错误: {str(e)[:200]}]"

        return self._finalize_response(reply)

    def stream_run(self, user_input: str) -> Generator[str, None, None]:
        user_message = Message(role="user", content=user_input)
        self.memory.add_message(user_message)
        self._start_scratchpad(user_input)

        if self.context_compressor is not None:
            self._check_and_compress()

        if self.planning_manager is not None:
            self._maybe_start_plan(user_input)

        history = self.memory.get_recent_messages()
        rag_results = self._get_rag()
        planning_context = self._get_planning_context()

        tool_results = None
        if self.tool_fn:
            try:
                tool_result = self.tool_fn(user_input)
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

        full_reply = ""
        try:
            for chunk in self.llm.stream_generate(messages=context):
                full_reply += chunk
                yield chunk
        except Exception as e:
            error_msg = f"[生成失败: {e}]"
            full_reply = error_msg
            yield error_msg

        assistant_message = Message(role="assistant", content=full_reply)
        self.memory.add_message(assistant_message)
        self._maybe_auto_summarize()

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
        history = self.memory.get_recent_messages()
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
        lowered = str(text or "").lower()
        phrases = (
            "用实际代码",
            "看看相关",
            "查看相关",
            "先看看",
            "读取",
            "read_file",
            "tool_calls",
            "invoke",
            "dsml",
        )
        return any(phrase in lowered for phrase in phrases)

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
        return "这个问题属于普通问答，本轮没有实际调用工具。我会直接根据已有上下文用自然语言回答。"

    def _finalize_response(self, reply: str) -> str:
        reply = self._sanitize_plain_reply(reply)
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
