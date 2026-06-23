"""Structured runtime events for Web/API integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


USER_MESSAGE = "user_message"
ASSISTANT_TEXT = "assistant_text"
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
PLANNING_UPDATE = "planning_update"
TODO_UPDATE = "todo_update"
PERMISSION_REQUEST = "permission_request"
FINAL = "final"
ERROR = "error"
DONE = "done"


@dataclass
class AgentEvent:
    """A serializable event emitted while the agent is running."""

    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp,
        }
        if self.session_id:
            payload["session_id"] = self.session_id
        return payload


def make_event(
    event_type: str,
    data: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> AgentEvent:
    return AgentEvent(type=event_type, data=data or {}, session_id=session_id)
