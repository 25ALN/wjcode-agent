"""Runtime orchestration and session primitives."""

from core.runtime.message import Message
from core.runtime.events import (
    AgentEvent,
    USER_MESSAGE,
    ASSISTANT_TEXT,
    TOOL_CALL,
    TOOL_RESULT,
    PLANNING_UPDATE,
    TODO_UPDATE,
    PERMISSION_REQUEST,
    FINAL,
    ERROR,
    DONE,
    make_event,
)

MAX_TOOL_ROUNDS = 12

__all__ = [
    "AgentRuntime",
    "MAX_TOOL_ROUNDS",
    "Message",
    "AgentEvent",
    "USER_MESSAGE",
    "ASSISTANT_TEXT",
    "TOOL_CALL",
    "TOOL_RESULT",
    "PLANNING_UPDATE",
    "TODO_UPDATE",
    "PERMISSION_REQUEST",
    "FINAL",
    "ERROR",
    "DONE",
    "make_event",
]


def __getattr__(name: str):
    if name == "AgentRuntime":
        from core.runtime.agent_runtime import AgentRuntime

        return AgentRuntime
    raise AttributeError(f"module 'core.runtime' has no attribute {name!r}")
