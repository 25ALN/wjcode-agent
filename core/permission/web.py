"""Permission primitives for Web/API driven approval flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
from uuid import uuid4

from core.permission import PermissionDecision, PermissionManager


class PermissionPending(Exception):
    """Raised when tool execution must pause for external approval."""

    def __init__(self, request: "PendingPermissionRequest"):
        super().__init__(f"Permission pending: {request.request_id}")
        self.request = request


@dataclass
class PendingPermissionRequest:
    request_id: str
    tool_name: str
    args: dict
    decision: PermissionDecision
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    resolved: bool = False
    approved: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "args": self.args,
            "risk_level": self.decision.risk_level,
            "reason": self.decision.reason,
            "allowed": self.decision.allowed,
            "requires_approval": self.decision.requires_approval,
            "created_at": self.created_at,
            "resolved": self.resolved,
            "approved": self.approved,
        }


class WebPermissionController:
    """Turns synchronous permission checks into pause/resume requests."""

    def __init__(self, manager: PermissionManager):
        self.manager = manager
        self._pending: Dict[str, PendingPermissionRequest] = {}

    @property
    def pending(self) -> Optional[PendingPermissionRequest]:
        for request in self._pending.values():
            if not request.resolved:
                return request
        return None

    def approve(self, tool_name: str, args: dict) -> bool:
        decision = self.manager.decide(tool_name, args)
        if not decision.allowed:
            return False
        if not decision.requires_approval:
            return True

        request = PendingPermissionRequest(
            request_id=uuid4().hex,
            tool_name=tool_name,
            args=dict(args),
            decision=decision,
        )
        self._pending[request.request_id] = request
        raise PermissionPending(request)

    def resolve(self, request_id: str, approved: bool) -> bool:
        request = self._pending.get(request_id)
        if request is None or request.resolved:
            return False
        request.resolved = True
        request.approved = bool(approved)
        return True

    def get(self, request_id: str) -> Optional[PendingPermissionRequest]:
        return self._pending.get(request_id)

    def clear_resolved(self) -> None:
        self._pending = {
            rid: request
            for rid, request in self._pending.items()
            if not request.resolved
        }
