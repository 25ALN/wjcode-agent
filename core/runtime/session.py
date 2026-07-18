"""Session management primitives for future Web/API servers."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from core.compression import ContextCompressor
from core.events import AgentEvent, ERROR, FINAL, USER_MESSAGE
from core.memory import LongMemory, ShortMemory
from core.message import Message
from core.permission import PermissionManager
from core.planning import PlanningManager
from core.project_context import ProjectContext
from core.runtime import AgentRuntime
from core.todo_store import TodoStore
from core.web_permission import WebPermissionController
from tools.code_executor import CodeExecutorTool
from tools.file_tool import FileReadTool, FileWriteTool
from tools.project_tools import EditTool, GrepTool, LSTool
from tools.registry import ToolRegistry
from tools.todo_tool import TodoUpdateTool
from tools.web_search import WebSearchTool


DEFAULT_SYSTEM_PROMPT = (
    "你是一个 AI 编程助手。简单任务使用 ReAct，复杂任务使用 Planning + ReAct。"
    "只有当问题需要查看、修改或验证当前项目时才使用工具；概念、建议和普通问答直接回答，且禁止输出 DSML/tool_calls 等工具协议文本。代码理解优先使用 ls/grep/read_file，小范围修改优先使用 edit_file。"
)

HISTORY_FILENAME = "history.json"
TOKEN_USAGE_KEYS = (
    "requests",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "estimated_prompt_tokens",
    "estimated_completion_tokens",
    "estimated_total_tokens",
)


def _now() -> float:
    return datetime.now().timestamp()


def _system_prompt_with_current_date(prompt: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return (
        f"{prompt}\n\n"
        f"【当前日期】{today}。涉及今天、明天、后天等相对日期时，以该日期为基准；"
        "实时天气、新闻、价格等信息应使用联网搜索工具核验。"
    )


@dataclass
class AgentSession:
    session_id: str
    runtime: AgentRuntime
    workspace_root: str
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    permission_controller: Optional[WebPermissionController] = None
    storage_dir: str = ""
    history_path: str = ""
    title: str = ""
    history_messages: list[dict] = field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = _now()

    def record_event(self, event: AgentEvent) -> bool:
        role = None
        content = None
        if event.type == USER_MESSAGE:
            role = "user"
            content = event.data.get("content")
        elif event.type == FINAL:
            role = "assistant"
            content = event.data.get("content")
        elif event.type == ERROR:
            role = "error"
            content = event.data.get("message")

        content = str(content or "").strip()
        if not role or not content:
            return False

        self.history_messages.append({
            "id": uuid4().hex,
            "role": role,
            "content": content,
            "timestamp": event.timestamp,
        })
        if role == "user" and not self.title:
            self.title = self._make_title(content)
        self.touch()
        self.save_history()
        return True

    def save_history(self) -> None:
        if not self.history_path:
            return
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        payload = {
            "session_id": self.session_id,
            "workspace_root": self.workspace_root,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "title": self.title or self._derive_title(),
            "messages": self.history_messages,
        }
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def snapshot(self, include_messages: bool = True) -> dict:
        visible_messages = self.visible_messages()
        if visible_messages and not self.history_messages:
            self.history_messages = visible_messages
            if not self.title:
                self.title = self._derive_title()
            self.save_history()

        planning = None
        if self.runtime.planning_manager is not None:
            planning = self.runtime.planning_manager.format_for_prompt()
        todo = None
        todo_progress = 0.0
        if self.runtime.todo_list is not None:
            todo = self.runtime.todo_list.format_for_prompt()
            if hasattr(self.runtime.todo_list, "progress"):
                todo_progress = self.runtime.todo_list.progress()
        pending = None
        if self.permission_controller is not None and self.permission_controller.pending:
            pending = self.permission_controller.pending.to_dict()
        scratchpad = None
        if getattr(self.runtime, "scratchpad", None) is not None:
            scratchpad = self.runtime.scratchpad.to_dict()

        payload = {
            "session_id": self.session_id,
            "workspace_root": self.workspace_root,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "title": self.title or self._derive_title(),
            "last_message": self._last_message_preview(visible_messages),
            "memory_messages": len(self.runtime.memory.messages),
            "history_messages": len(visible_messages),
            "planning": planning,
            "todo": todo,
            "todo_progress": todo_progress,
            "pending_permission": pending,
            "scratchpad": scratchpad,
        }
        if include_messages:
            payload["messages"] = visible_messages
        return payload

    def visible_messages(self, limit: int = 300) -> list[dict]:
        if self.history_messages:
            return [dict(item) for item in self.history_messages[-limit:]]
        return self._visible_messages_from_runtime(limit)

    def _derive_title(self) -> str:
        for item in self.history_messages:
            if item.get("role") == "user" and item.get("content"):
                return self._make_title(str(item["content"]))
        return self.title or "新会话"

    def _last_message_preview(self, messages: Optional[list[dict]] = None) -> str:
        source = messages if messages is not None else self.history_messages
        for item in reversed(source):
            content = str(item.get("content") or "").strip()
            if content:
                return self._make_title(content, limit=56)
        return ""

    def _visible_messages_from_runtime(self, limit: int = 300) -> list[dict]:
        visible = []
        for msg in self.runtime.memory.get_recent_messages():
            if msg.role not in {"user", "assistant"}:
                continue
            metadata = msg.metadata or {}
            if msg.role == "assistant" and (
                metadata.get("function_calls") or metadata.get("function_call")
            ):
                continue
            content = str(msg.content or "").strip()
            if not content:
                continue
            visible.append({
                "id": uuid4().hex,
                "role": msg.role,
                "content": content,
                "timestamp": float(msg.timestamp or _now()),
            })
        return visible[-limit:]

    @staticmethod
    def _make_title(content: str, limit: int = 28) -> str:
        text = " ".join(str(content or "").split())
        if len(text) <= limit:
            return text or "新会话"
        return text[:limit].rstrip() + "..."


class AgentSessionManager:
    """Creates isolated AgentRuntime instances for Web/API use."""

    def __init__(
        self,
        llm_factory: Callable[[], object],
        workspace_root: Optional[str] = None,
        storage_root: Optional[str] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        enable_tools: bool = True,
        enable_permissions: bool = True,
        memory_embedder: Optional[Any] = None,
    ):
        self.llm_factory = llm_factory
        self.workspace_root = os.path.abspath(workspace_root or os.getcwd())
        self.storage_root = os.path.abspath(storage_root or os.path.join(self.workspace_root, ".agent_sessions"))
        self.system_prompt = system_prompt
        self.enable_tools = enable_tools
        self.enable_permissions = enable_permissions
        self.memory_embedder = memory_embedder
        self._sessions: Dict[str, AgentSession] = {}
        self._archived_api_usage = self._empty_api_usage()

    def create_session(self, session_id: Optional[str] = None) -> AgentSession:
        sid = session_id or uuid4().hex
        if sid in self._sessions or self._history_exists(sid):
            raise ValueError(f"session 已存在: {sid}")

        session = self._build_session(sid, history_payload=None)
        self._sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        session = self._sessions.get(session_id)
        if session is None and self._history_exists(session_id):
            session = self._build_session(session_id, self._load_history(session_id))
            self._sessions[session_id] = session
        if session:
            session.touch()
        return session

    def close_session(self, session_id: str) -> bool:
        return self.delete_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        existed = session is not None
        if session is not None:
            self._merge_api_usage(
                self._archived_api_usage,
                self._runtime_token_usage(session).get("api") or {},
            )
        session_dir = self._session_dir(session_id)
        if os.path.isdir(session_dir):
            shutil.rmtree(session_dir, ignore_errors=True)
            existed = True
        return existed

    def get_token_usage_summary(self) -> dict:
        totals = dict(self._archived_api_usage)
        sessions = []
        for sid, session in self._sessions.items():
            usage = self._runtime_token_usage(session)
            api_usage = usage.get("api") or {}
            self._merge_api_usage(totals, api_usage)
            sessions.append({
                "session_id": sid,
                "title": session.title or session._derive_title(),
                "memory_messages": usage.get("memory_messages") or 0,
                "context_tokens": usage.get("context_tokens") or 0,
                "api": dict(api_usage),
            })
        return {
            "active_sessions": len(self._sessions),
            "archived_api": dict(self._archived_api_usage),
            "sessions": sessions,
            "totals": totals,
        }

    def list_sessions(self) -> list[dict]:
        snapshots: Dict[str, dict] = {
            sid: session.snapshot(include_messages=False)
            for sid, session in self._sessions.items()
        }
        for sid in self._stored_session_ids():
            if sid not in snapshots:
                payload = self._load_history(sid)
                if payload is not None and self._history_messages(payload):
                    snapshots[sid] = self._snapshot_from_history_payload(sid, payload)
        return sorted(
            snapshots.values(),
            key=lambda item: item.get("updated_at") or item.get("created_at") or 0,
            reverse=True,
        )

    def _build_session(self, sid: str, history_payload: Optional[dict]) -> AgentSession:
        session_dir = self._session_dir(sid)
        todo_store = TodoStore(os.path.join(session_dir, "todo.json"))
        todo = todo_store.load()
        registry = self._build_registry(todo, todo_store, self.workspace_root) if self.enable_tools else None
        permission_controller = None
        permission_fn = None

        if registry is not None and self.enable_permissions:
            permission_manager = PermissionManager(
                tool_registry=registry,
                project_root=self.workspace_root,
            )
            permission_controller = WebPermissionController(permission_manager)
            permission_fn = permission_controller.approve

        project_context = ProjectContext(os.path.join(self.workspace_root, "AGENT.md")).get_context_str()
        planning_manager = PlanningManager(todo_list=todo, enable_llm_planning=False)
        long_memory = LongMemory(
            storage_path=os.path.join(session_dir, "memory_long.json"),
            embedder=self.memory_embedder,
        )
        context_compressor = ContextCompressor(threshold_tokens=50000, keep_recent=12)
        history_messages = self._history_messages(history_payload)
        memory = ShortMemory(max_length=60)
        memory.replace_messages(self._memory_from_history(history_messages))

        runtime = AgentRuntime(
            llm_client=self.llm_factory(),
            memory=memory,
            long_memory=long_memory,
            system_prompt=_system_prompt_with_current_date(self.system_prompt),
            tool_registry=registry,
            project_context=project_context,
            todo_list=todo,
            planning_manager=planning_manager,
            permission_fn=permission_fn,
            context_compressor=context_compressor,
            session_id=sid,
        )
        runtime.todo_store = todo_store

        created_at = float(history_payload.get("created_at", _now())) if history_payload else _now()
        updated_at = float(history_payload.get("updated_at", created_at)) if history_payload else created_at
        title = str(history_payload.get("title") or "") if history_payload else ""

        return AgentSession(
            session_id=sid,
            runtime=runtime,
            workspace_root=self.workspace_root,
            created_at=created_at,
            updated_at=updated_at,
            permission_controller=permission_controller,
            storage_dir=session_dir,
            history_path=self._history_path(sid),
            title=title,
            history_messages=history_messages,
        )

    def _session_dir(self, session_id: str) -> str:
        return os.path.join(self.storage_root, session_id)

    def _history_path(self, session_id: str) -> str:
        return os.path.join(self._session_dir(session_id), HISTORY_FILENAME)

    def _history_exists(self, session_id: str) -> bool:
        return os.path.isfile(self._history_path(session_id))

    def _stored_session_ids(self) -> list[str]:
        if not os.path.isdir(self.storage_root):
            return []
        result = []
        for name in os.listdir(self.storage_root):
            if os.path.isfile(os.path.join(self.storage_root, name, HISTORY_FILENAME)):
                result.append(name)
        return result

    def _load_history(self, session_id: str) -> Optional[dict]:
        path = self._history_path(session_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload if isinstance(payload, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _history_messages(history_payload: Optional[dict]) -> list[dict]:
        raw = history_payload.get("messages", []) if history_payload else []
        if not isinstance(raw, list):
            return []
        messages = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = str(item.get("content") or "").strip()
            if role not in {"user", "assistant", "error"} or not content:
                continue
            messages.append({
                "id": str(item.get("id") or uuid4().hex),
                "role": role,
                "content": content,
                "timestamp": float(item.get("timestamp") or _now()),
            })
        return messages

    @staticmethod
    def _memory_from_history(history_messages: list[dict]) -> list[Message]:
        messages = []
        for item in history_messages:
            if item.get("role") not in {"user", "assistant"}:
                continue
            messages.append(Message(
                role=item["role"],
                content=item["content"],
                timestamp=float(item.get("timestamp") or _now()),
            ))
        return messages

    def _snapshot_from_history_payload(self, sid: str, payload: dict) -> dict:
        messages = self._history_messages(payload)
        title = str(payload.get("title") or "") or self._title_from_messages(messages)
        updated_at = float(payload.get("updated_at") or self._last_timestamp(messages) or _now())
        created_at = float(payload.get("created_at") or updated_at)
        return {
            "session_id": sid,
            "workspace_root": str(payload.get("workspace_root") or self.workspace_root),
            "created_at": created_at,
            "updated_at": updated_at,
            "title": title,
            "last_message": self._last_preview_from_messages(messages),
            "memory_messages": len([m for m in messages if m.get("role") in {"user", "assistant"}]),
            "history_messages": len(messages),
            "planning": None,
            "todo": None,
            "todo_progress": 0.0,
            "pending_permission": None,
            "scratchpad": None,
        }

    @staticmethod
    def _title_from_messages(messages: list[dict]) -> str:
        for item in messages:
            if item.get("role") == "user":
                return AgentSession._make_title(str(item.get("content") or ""))
        return "新会话"

    @staticmethod
    def _last_preview_from_messages(messages: list[dict]) -> str:
        for item in reversed(messages):
            content = str(item.get("content") or "").strip()
            if content:
                return AgentSession._make_title(content, limit=56)
        return ""

    @staticmethod
    def _last_timestamp(messages: list[dict]) -> float:
        for item in reversed(messages):
            if item.get("timestamp"):
                return float(item["timestamp"])
        return 0.0

    @staticmethod
    def _empty_api_usage() -> dict:
        return {key: 0 for key in TOKEN_USAGE_KEYS}

    @staticmethod
    def _merge_api_usage(target: dict, source: dict) -> None:
        for key in TOKEN_USAGE_KEYS:
            target[key] = int(target.get(key) or 0) + int(source.get(key) or 0)

    @staticmethod
    def _runtime_token_usage(session: AgentSession) -> dict:
        runtime = session.runtime
        if hasattr(runtime, "get_token_usage_summary"):
            try:
                return runtime.get_token_usage_summary()
            except Exception:
                pass
        memory = getattr(runtime, "memory", None)
        return {
            "memory_messages": len(getattr(memory, "messages", []) or []),
            "context_tokens": memory.total_tokens() if hasattr(memory, "total_tokens") else 0,
            "api": {},
        }

    @staticmethod
    def _build_registry(todo, todo_store: TodoStore, workspace_root: str) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(FileReadTool(workspace_root=workspace_root))
        registry.register(FileWriteTool(workspace_root=workspace_root))
        registry.register(CodeExecutorTool())
        registry.register(WebSearchTool())
        registry.register(LSTool(workspace_root=workspace_root))
        registry.register(GrepTool(workspace_root=workspace_root))
        registry.register(EditTool(workspace_root=workspace_root))
        registry.register(TodoUpdateTool(todo, todo_store))
        return registry
