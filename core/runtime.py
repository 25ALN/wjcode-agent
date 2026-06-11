import logging
from typing import List, Optional, Generator, Callable, Any, Dict
from core.message import Message
from core.memory import ShortMemory, LongMemory
from core.context import build_context
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 12  # 复杂任务（如分析多文件目录）需要更多工具轮次


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
        permission_fn: Optional[Callable] = None,
        context_compressor: Optional[object] = None,
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
        self.permission_fn = permission_fn
        self.context_compressor = context_compressor

    def run(self, user_input: str) -> str:
        user_message = Message(role="user", content=user_input)
        self.memory.add_message(user_message)

        if self.context_compressor is not None:
            self._check_and_compress()

        if self.tool_registry and self.tool_registry.tool_count > 0:
            return self._run_with_native_tools()

        return self._run_with_callback_tool()

    def _run_with_native_tools(self) -> str:
        rag_results = self._get_rag()
        long_memory_context = self._get_long_memory_context()
        todo_context = self._get_todo_context()

        context = self._build_context(
            rag_results=rag_results,
            long_memory_context=long_memory_context,
            todo_context=todo_context,
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

        context = build_context(
            system_prompt=self.system_prompt,
            messages=history,
            rag_results=rag_results,
            tool_results=tool_results,
            long_memory_context=long_memory_context,
            project_context=self.project_context,
            todo_context=todo_context,
        )

        try:
            response = self.llm.generate(messages=context)
            reply = response if isinstance(response, str) else response.get("text", str(response))
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            reply = f"[抱歉，我遇到了一个错误: {str(e)[:200]}]"

        return self._finalize_response(reply)
    def stream_run(self, user_input: str) -> Generator[str, None, None]:
        user_message = Message(role="user", content=user_input)
        self.memory.add_message(user_message)

        history = self.memory.get_recent_messages()
        rag_results = self._get_rag()

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

        context = build_context(
            system_prompt=self.system_prompt,
            messages=history,
            rag_results=rag_results,
            tool_results=tool_results,
            long_memory_context=long_memory_context,
            project_context=self.project_context,
            todo_context=todo_context,
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
            return self.long_memory.get_context()
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
    ) -> List[Message]:
        history = self.memory.get_recent_messages()
        return build_context(
            system_prompt=self.system_prompt,
            messages=history,
            rag_results=rag_results,
            long_memory_context=long_memory_context,
            project_context=self.project_context,
            todo_context=todo_context,
        )

    def _execute_tool(self, name: str, args: dict) -> str:
        if not isinstance(args, dict):
            return f"[错误] 工具参数必须是 JSON 对象，实际收到: {type(args).__name__}"

        if self.permission_fn is not None:
            try:
                approved = self.permission_fn(name, args)
                if not approved:
                    return f"[权限拒绝] Tool '{name}' 需要用户确认，但未获批准。"
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

    def _finalize_response(self, reply: str) -> str:
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
                logger.info(f"上下文压缩完成: {len(compression_result) if compression_result else 0} 字符")
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
        if self.tool_registry:
            parts.append(f"可用工具: {self.tool_registry.list_names()}")
        if self.project_context:
            parts.append(f"AGENT.md: 已加载 ({len(self.project_context)} 字符)")
        if self.todo_list:
            parts.append(f"TodoList: 已配置")
        return "\n".join(parts)

    def reset_memory(self):
        self.memory.clear()
        self._conversation_turns = 0

    def reset_long_memory(self):
        if self.long_memory:
            self.long_memory.clear()

    def reset_all(self):
        self.reset_memory()
        self.reset_long_memory()
