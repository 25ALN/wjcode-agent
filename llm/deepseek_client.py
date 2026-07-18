import os
import time
import json
import logging
import urllib.request
import urllib.error
from typing import List, Optional, Generator, Dict, Any

from dotenv import load_dotenv
from core.message import Message

load_dotenv()

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1/chat/completions"


class DeepSeekClient:

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        retry_times: int = 3,
        retry_delay: float = 1.0,
        timeout: Optional[int] = 60,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY 未设置。请在 .env 文件中添加：\n"
                "DEEPSEEK_API_KEY=你的密钥"
            )
        self.model_name = model_name
        self.retry_times = retry_times
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self._usage = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_prompt_tokens": 0,
            "estimated_completion_tokens": 0,
            "estimated_total_tokens": 0,
        }

    def generate(
        self,
        messages: List[Message],
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> dict:
        """调用 DeepSeek Chat Completions API，支持 Function Calling"""
        payload = self._build_payload(messages, system_instruction, tools)
        logger.debug(
            f"DeepSeek 请求: {len(messages)} 条消息, "
            f"tools={bool(tools)}, model={self.model_name}"
        )
        estimated_prompt_tokens = self._estimate_payload_tokens(payload)
        response = self._call_with_retry(payload)
        parsed = self._parse_response(response)
        estimated_completion_tokens = Message.estimate_tokens(parsed.get("text") or "")
        self._record_usage(
            response,
            estimated_prompt_tokens=estimated_prompt_tokens,
            estimated_completion_tokens=estimated_completion_tokens,
        )
        return parsed

    def stream_generate(
        self,
        messages: List[Message],
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[str, None, None]:
        """流式生成。简化处理为纯文本流。"""
        payload = self._build_payload(messages, system_instruction, tools)
        payload["stream"] = True
        try:
            for chunk_text in self._call_stream(payload):
                yield chunk_text
        except Exception as e:
            logger.error(f"DeepSeek 流式生成失败: {e}")
            yield f"[生成失败: {str(e)[:100]}]"

    def count_tokens(self, messages: List[Message]) -> int:
        return sum(msg.token_count() for msg in messages)

    def get_usage_summary(self) -> dict:
        """Return accumulated API token usage for the current client instance."""
        return dict(self._usage)

    def reset_usage_summary(self) -> None:
        for key in self._usage:
            self._usage[key] = 0

    def check_available(self) -> bool:
        try:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            }
            self._call_api(payload, timeout=10)
            return True
        except Exception as e:
            logger.warning(f"DeepSeek API 不可用: {e}")
            return False

    # ── 请求构建 ──────────────────────────────

    def _build_payload(
        self,
        messages: List[Message],
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> dict:
        payload = {
            "model": self.model_name,
            "messages": self._messages_to_openai_format(messages, system_instruction),
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.max_output_tokens is not None:
            payload["max_tokens"] = self.max_output_tokens

        if tools:
            openai_tools = self._normalize_tools(tools)
            if openai_tools:
                payload["tools"] = openai_tools
                payload["tool_choice"] = "auto"

        return payload

    def _normalize_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for tool in tools:
            # 已经是 OpenAI 格式
            if "type" in tool and tool.get("type") == "function":
                result.append(tool)
                continue

            if "name" in tool:
                result.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    },
                })
                continue

            # 未知格式，尝试原样传入
            logger.warning(f"未知的 tool 格式，直接传入: {list(tool.keys())}")
            result.append(tool)

        return result

    def _messages_to_openai_format(
        self,
        messages: List[Message],
        system_instruction: Optional[str] = None,
    ) -> List[dict]:
        """将内部 Message 列表转为 OpenAI Chat Completions 格式"""
        result = []

        # 系统指令
        sys_content = system_instruction
        if not sys_content:
            sys_parts = [
                m.content for m in messages
                if m.role == "system" and not m.metadata
            ]
            if sys_parts:
                sys_content = "\n\n".join(sys_parts)

        if sys_content:
            result.append({"role": "system", "content": sys_content})

        # 非 system 消息
        for msg in messages:
            if msg.role == "system":
                continue

            metadata = msg.metadata or {}

            # assistant 消息带 function_call/function_calls
            function_calls = metadata.get("function_calls")
            if not function_calls and metadata.get("function_call"):
                function_calls = [metadata["function_call"]]

            if msg.role == "assistant" and function_calls:
                tool_calls = []
                for idx, fc in enumerate(function_calls):
                    call_id = fc.get("id") or self._fallback_tool_call_id(fc, idx)
                    tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": fc.get("name", ""),
                            "arguments": json.dumps(fc.get("args", {}), ensure_ascii=False),
                        },
                    })

                result.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": tool_calls,
                })
                continue

            # tool 消息带 function_response
            if msg.role == "tool" and metadata.get("function_response"):
                fr = metadata["function_response"]
                result.append({
                    "role": "tool",
                    "tool_call_id": fr.get("id") or self._fallback_tool_call_id(fr, 0),
                    "content": fr.get("response", {}).get("result", msg.content),
                })
                continue

            # 普通消息
            result.append({
                "role": msg.role if msg.role in ("user", "assistant") else "user",
                "content": msg.content,
            })

        return result

    def _record_usage(
        self,
        response: dict,
        estimated_prompt_tokens: int = 0,
        estimated_completion_tokens: int = 0,
    ) -> None:
        self._usage["requests"] += 1
        self._usage["estimated_prompt_tokens"] += int(estimated_prompt_tokens or 0)
        self._usage["estimated_completion_tokens"] += int(estimated_completion_tokens or 0)
        self._usage["estimated_total_tokens"] += int(estimated_prompt_tokens or 0) + int(estimated_completion_tokens or 0)

        usage = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(usage, dict):
            return

        prompt_tokens = self._usage_int(usage, "prompt_tokens", "input_tokens")
        completion_tokens = self._usage_int(usage, "completion_tokens", "output_tokens")
        total_tokens = self._usage_int(usage, "total_tokens")
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens

        self._usage["prompt_tokens"] += prompt_tokens
        self._usage["completion_tokens"] += completion_tokens
        self._usage["total_tokens"] += total_tokens

    @staticmethod
    def _usage_int(usage: dict, *keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        return 0

    @staticmethod
    def _estimate_payload_tokens(payload: dict) -> int:
        messages = payload.get("messages", []) if isinstance(payload, dict) else []
        total = 0
        for item in messages:
            if not isinstance(item, dict):
                continue
            total += Message.estimate_tokens(str(item.get("role") or ""))
            total += Message.estimate_tokens(str(item.get("content") or ""))
            tool_calls = item.get("tool_calls")
            if tool_calls:
                total += Message.estimate_tokens(json.dumps(tool_calls, ensure_ascii=False))
        tools = payload.get("tools")
        if tools:
            total += Message.estimate_tokens(json.dumps(tools, ensure_ascii=False))
        return total

    def _parse_response(self, response: dict) -> dict:
        result = {"text": None, "function_call": None, "function_calls": []}
        try:
            choice = response["choices"][0]
            msg = choice.get("message", {})

            text = msg.get("content")
            if text:
                result["text"] = text.strip()

            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                parsed_calls = []
                for idx, tc in enumerate(tool_calls):
                    fn_info = tc.get("function", {})
                    fn_name = fn_info.get("name", "")
                    fn_args = self._parse_tool_arguments(fn_info.get("arguments", "{}"))
                    parsed_calls.append({
                        "id": tc.get("id") or self._fallback_tool_call_id(
                            {"name": fn_name}, idx
                        ),
                        "name": fn_name,
                        "args": fn_args,
                    })

                result["function_calls"] = parsed_calls
                result["function_call"] = parsed_calls[0] if parsed_calls else None

        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"解析 DeepSeek 响应失败: {e}")

        logger.debug(
            f"DeepSeek 响应: text={bool(result['text'])}, "
            f"fn_calls={len(result['function_calls'])}"
        )
        return result

    @staticmethod
    def _parse_tool_arguments(arguments: Any) -> dict:
        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str):
            return {}
        try:
            parsed = json.loads(arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"DeepSeek 参数非 JSON: {arguments[:200]}")
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _fallback_tool_call_id(call: dict, index: int) -> str:
        name = call.get("name", "tool")
        return f"call_{index}_{abs(hash(name))}"

    def _call_api(self, payload: dict, timeout: Optional[int] = None) -> dict:
        timeout = timeout or self.timeout or 60
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            DEEPSEEK_BASE_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"DeepSeek API HTTP {e.code}: {error_body[:300]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"DeepSeek API 网络错误: {e}") from e

    def _call_with_retry(self, payload: dict) -> dict:
        last_error = None
        delay = self.retry_delay
        for attempt in range(1, self.retry_times + 1):
            try:
                return self._call_api(payload)
            except Exception as e:
                last_error = e
                error_str = str(e)
                logger.warning(f"DeepSeek API 调用失败 (第{attempt}次): {error_str[:100]}")
                if "401" in error_str or "403" in error_str:
                    raise
                if attempt < self.retry_times:
                    time.sleep(delay)
                    delay *= 2
        raise RuntimeError(
            f"DeepSeek API 在 {self.retry_times} 次重试后仍然失败: {last_error}"
        ) from last_error

    def _call_stream(self, payload: dict) -> Generator[str, None, None]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            DEEPSEEK_BASE_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout or 60) as resp:
                for line in resp:
                    line = line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except Exception as e:
            logger.error(f"DeepSeek 流式读取失败: {e}")
            yield f"[流式生成失败: {str(e)[:100]}]"
