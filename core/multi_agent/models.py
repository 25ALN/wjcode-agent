"""Data contracts for the sequential multi-agent workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class AgentRole(str, Enum):
    COORDINATOR = "coordinator"
    EXPLORE = "explore"
    IMPLEMENT = "implement"
    TEST = "test"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_PERMISSION = "waiting_permission"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentTask:
    """A bounded unit of work given to one specialised agent."""

    role: AgentRole
    instruction: str
    acceptance_criteria: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid4().hex)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional["AgentTaskResult"] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role.value,
            "instruction": self.instruction,
            "acceptance_criteria": list(self.acceptance_criteria),
            "status": self.status.value,
            "result": self.result.to_dict() if self.result else None,
        }


@dataclass
class AgentTaskResult:
    """Small, serialisable result passed to the next role and coordinator."""

    status: TaskStatus
    summary: str = ""
    evidence: List[str] = field(default_factory=list)
    next_action: str = "finish"
    changes: List[str] = field(default_factory=list)
    verification: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "summary": self.summary,
            "evidence": list(self.evidence),
            "next_action": self.next_action,
            "changes": list(self.changes),
            "verification": list(self.verification),
            "risks": list(self.risks),
            "error": self.error,
        }
