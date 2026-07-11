"""Framework-independent Web service primitives.

This module adapts AgentSessionManager and AgentRuntime event generators into
operations that are easy to expose through HTTP/SSE, without requiring FastAPI
at import time. Tests can use this layer directly with fake LLM clients.
"""

from __future__ import annotations

import json
from typing import Iterable, Iterator, Optional

from core.events import AgentEvent, DONE, ERROR, make_event
from core.session import AgentSession, AgentSessionManager
from core.web_permission import PendingPermissionRequest


class AgentWebServiceError(Exception):
    """Base class for Web service layer errors."""


class SessionNotFoundError(AgentWebServiceError):
    """Raised when a session id is not known by the service."""


class PermissionControllerUnavailable(AgentWebServiceError):
    """Raised when a session has no permission controller configured."""


class PermissionRequestNotFoundError(AgentWebServiceError):
    """Raised when a permission request id cannot be resumed."""


class InvalidRequestError(AgentWebServiceError, ValueError):
    """Raised for malformed service requests."""


def sse_encode(event: AgentEvent) -> str:
    """Encode an AgentEvent as one Server-Sent Events frame."""
    payload = event.to_dict()
    event_type = str(payload.get("type") or "message")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_type}\n" + f"data: {data}\n\n"


def iter_sse(events: Iterable[AgentEvent]) -> Iterator[str]:
    """Yield Server-Sent Events frames for an AgentEvent iterable."""
    for event in events:
        yield sse_encode(event)


class AgentWebService:
    """Thin service layer around AgentSessionManager.

    The service owns no runtime behavior itself. It validates session and
    permission ids, then delegates to AgentRuntime.run_events()/resume_events().
    """

    def __init__(self, session_manager: AgentSessionManager):
        self.session_manager = session_manager

    def create_session(self, session_id: Optional[str] = None) -> dict:
        session = self.session_manager.create_session(session_id=session_id)
        return session.snapshot()

    def list_sessions(self) -> list[dict]:
        return self.session_manager.list_sessions()

    def get_session(self, session_id: str) -> AgentSession:
        session = self.session_manager.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"session not found: {session_id}")
        return session

    def get_snapshot(self, session_id: str) -> dict:
        return self.get_session(session_id).snapshot()

    def close_session(self, session_id: str) -> bool:
        if not self.session_manager.close_session(session_id):
            raise SessionNotFoundError(f"session not found: {session_id}")
        return True

    def pending_permission(self, session_id: str) -> Optional[dict]:
        session = self.get_session(session_id)
        controller = session.permission_controller
        if controller is None or controller.pending is None:
            return None
        return controller.pending.to_dict()

    def run_message(self, session_id: str, content: str) -> Iterator[AgentEvent]:
        session = self.get_session(session_id)
        message = self._validate_content(content)
        session.touch()
        yield from session.runtime.run_events(message)

    def run_message_sse(self, session_id: str, content: str) -> Iterator[str]:
        try:
            yield from iter_sse(self.run_message(session_id, content))
        except AgentWebServiceError:
            raise
        except Exception as exc:
            yield sse_encode(make_event(
                ERROR,
                {"message": f"Agent 执行失败: {str(exc)[:200]}"},
                session_id=session_id,
            ))
            yield sse_encode(make_event(DONE, {}, session_id=session_id))

    def resolve_permission(
        self,
        session_id: str,
        request_id: str,
        approved: bool,
    ) -> Iterator[AgentEvent]:
        session, request = self._get_pending_request(session_id, request_id)
        if not session.permission_controller.resolve(request.request_id, approved):
            raise PermissionRequestNotFoundError(
                f"permission request cannot be resolved: {request_id}"
            )
        session.touch()
        yield from session.runtime.resume_events(bool(approved))

    def resolve_permission_sse(
        self,
        session_id: str,
        request_id: str,
        approved: bool,
    ) -> Iterator[str]:
        try:
            yield from iter_sse(self.resolve_permission(session_id, request_id, approved))
        except AgentWebServiceError:
            raise
        except Exception as exc:
            yield sse_encode(make_event(
                ERROR,
                {"message": f"权限恢复执行失败: {str(exc)[:200]}"},
                session_id=session_id,
            ))
            yield sse_encode(make_event(DONE, {}, session_id=session_id))

    def error_event(self, message: str, session_id: Optional[str] = None) -> AgentEvent:
        return make_event(ERROR, {"message": message}, session_id=session_id)

    def _get_pending_request(
        self,
        session_id: str,
        request_id: str,
    ) -> tuple[AgentSession, PendingPermissionRequest]:
        session = self.get_session(session_id)
        controller = session.permission_controller
        if controller is None:
            raise PermissionControllerUnavailable(
                f"session has no permission controller: {session_id}"
            )
        request = controller.get(request_id)
        if request is None or request.resolved:
            raise PermissionRequestNotFoundError(
                f"permission request not found: {request_id}"
            )
        return session, request

    @staticmethod
    def _validate_content(content: str) -> str:
        message = str(content or "").strip()
        if not message:
            raise InvalidRequestError("message content is required")
        return message
