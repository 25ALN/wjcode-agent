"""Session management primitives for future Web/API servers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from core.compression import ContextCompressor
from core.memory import LongMemory, ShortMemory
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
    "代码理解优先使用 ls/grep/read_file，小范围修改优先使用 edit_file。"
)


@dataclass
class AgentSession:
    session_id: str
    runtime: AgentRuntime
    workspace_root: str
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = field(default_factory=lambda: datetime.now().timestamp())
    permission_controller: Optional[WebPermissionController] = None

    def touch(self) -> None:
        self.updated_at = datetime.now().timestamp()

    def snapshot(self) -> dict:
        planning = None
        if self.runtime.planning_manager is not None:
            planning = self.runtime.planning_manager.format_for_prompt()
        todo = None
        if self.runtime.todo_list is not None:
            todo = self.runtime.todo_list.format_for_prompt()
        pending = None
        if self.permission_controller is not None and self.permission_controller.pending:
            pending = self.permission_controller.pending.to_dict()
        return {
            "session_id": self.session_id,
            "workspace_root": self.workspace_root,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "memory_messages": len(self.runtime.memory.messages),
            "planning": planning,
            "todo": todo,
            "pending_permission": pending,
        }


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

    def create_session(self, session_id: Optional[str] = None) -> AgentSession:
        sid = session_id or uuid4().hex
        if sid in self._sessions:
            raise ValueError(f"session 已存在: {sid}")

        todo_store = TodoStore(os.path.join(self.storage_root, sid, "todo.json"))
        todo = todo_store.load()
        registry = self._build_registry(todo, todo_store) if self.enable_tools else None
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
            storage_path=os.path.join(self.storage_root, sid, "memory_long.json"),
            embedder=self.memory_embedder,
        )
        context_compressor = ContextCompressor(threshold_tokens=50000, keep_recent=12)

        runtime = AgentRuntime(
            llm_client=self.llm_factory(),
            memory=ShortMemory(max_length=60),
            long_memory=long_memory,
            system_prompt=self.system_prompt,
            tool_registry=registry,
            project_context=project_context,
            todo_list=todo,
            planning_manager=planning_manager,
            permission_fn=permission_fn,
            context_compressor=context_compressor,
            session_id=sid,
        )
        runtime.todo_store = todo_store

        session = AgentSession(
            session_id=sid,
            runtime=runtime,
            workspace_root=self.workspace_root,
            permission_controller=permission_controller,
        )
        self._sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        session = self._sessions.get(session_id)
        if session:
            session.touch()
        return session

    def close_session(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> list[dict]:
        return [session.snapshot() for session in self._sessions.values()]

    @staticmethod
    def _build_registry(todo, todo_store: TodoStore) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(FileReadTool())
        registry.register(FileWriteTool())
        registry.register(CodeExecutorTool())
        registry.register(WebSearchTool())
        registry.register(LSTool())
        registry.register(GrepTool())
        registry.register(EditTool())
        registry.register(TodoUpdateTool(todo, todo_store))
        return registry
