"""FastAPI adapter for the agent Web service.

Install optional Web dependencies before running this module:
    pip install fastapi uvicorn
"""

from __future__ import annotations

import os
from typing import Optional

try:
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, Response, StreamingResponse
    from pydantic import BaseModel, Field
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without deps
    Body = None
    FastAPI = None
    HTTPException = None
    CORSMiddleware = None
    FileResponse = None
    Response = None
    StreamingResponse = None
    BaseModel = object
    Field = None
    _FASTAPI_IMPORT_ERROR = exc
else:
    _FASTAPI_IMPORT_ERROR = None

from core.session import AgentSessionManager
from llm.deepseek_client import DeepSeekClient
from server.service import (
    AgentWebService,
    InvalidRequestError,
    PermissionControllerUnavailable,
    PermissionRequestNotFoundError,
    SessionNotFoundError,
)


if Field is not None:

    class CreateSessionRequest(BaseModel):
        session_id: Optional[str] = Field(default=None, min_length=1)


    class MessageRequest(BaseModel):
        content: str = Field(min_length=1)


    class PermissionDecisionRequest(BaseModel):
        approved: bool

else:  # pragma: no cover

    class CreateSessionRequest:  # type: ignore[no-redef]
        pass


    class MessageRequest:  # type: ignore[no-redef]
        pass


    class PermissionDecisionRequest:  # type: ignore[no-redef]
        pass


def create_default_service(
    workspace_root: Optional[str] = None,
    storage_root: Optional[str] = None,
) -> AgentWebService:
    """Create the default service backed by DeepSeek and built-in tools."""
    workspace = os.path.abspath(workspace_root or os.getcwd())
    storage = os.path.abspath(storage_root or os.path.join(workspace, ".agent_sessions"))
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    timeout = int(os.getenv("DEEPSEEK_TIMEOUT", "30"))
    retry_times = int(os.getenv("DEEPSEEK_RETRY_TIMES", "1"))

    def llm_factory():
        return DeepSeekClient(
            model_name=model_name,
            temperature=1,
            timeout=timeout,
            retry_times=retry_times,
        )

    manager = AgentSessionManager(
        llm_factory=llm_factory,
        workspace_root=workspace,
        storage_root=storage,
        enable_tools=True,
        enable_permissions=True,
    )
    return AgentWebService(manager)


def create_app(service: Optional[AgentWebService] = None):
    """Create a FastAPI app for the Agent Web API."""
    if FastAPI is None:
        raise RuntimeError(
            "FastAPI is not installed. Install Web dependencies with: "
            "pip install fastapi uvicorn"
        ) from _FASTAPI_IMPORT_ERROR

    app = FastAPI(title="Coder Agent API", version="0.3.0")
    app.state.agent_service = service or create_default_service()

    origins = [
        origin.strip()
        for origin in os.getenv(
            "AGENT_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8001,http://127.0.0.1:8001,http://localhost:8080,http://127.0.0.1:8080",
        ).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def svc() -> AgentWebService:
        return app.state.agent_service

    web_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))

    def web_file(name: str) -> str:
        return os.path.join(web_root, name)

    @app.get("/", include_in_schema=False)
    def frontend_index():
        index_path = web_file("index.html")
        if not os.path.exists(index_path):
            raise HTTPException(status_code=404, detail="web/index.html not found")
        return FileResponse(index_path, headers={"Cache-Control": "no-store"})

    @app.get("/app.js", include_in_schema=False)
    def frontend_script():
        script_path = web_file("app.js")
        if not os.path.exists(script_path):
            raise HTTPException(status_code=404, detail="web/app.js not found")
        return FileResponse(script_path, media_type="application/javascript", headers={"Cache-Control": "no-store"})

    @app.get("/styles.css", include_in_schema=False)
    def frontend_styles():
        style_path = web_file("styles.css")
        if not os.path.exists(style_path):
            raise HTTPException(status_code=404, detail="web/styles.css not found")
        return FileResponse(style_path, media_type="text/css", headers={"Cache-Control": "no-store"})

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    @app.get("/health")
    def health():
        return {"ok": True, "sessions": len(svc().list_sessions())}

    @app.post("/sessions")
    def create_session(payload: Optional[CreateSessionRequest] = Body(default=None)):
        try:
            session_id = payload.session_id if payload is not None else None
            return svc().create_session(session_id=session_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/sessions")
    def list_sessions():
        return {"sessions": svc().list_sessions()}

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str):
        try:
            return svc().get_snapshot(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/sessions/{session_id}")
    def close_session(session_id: str):
        try:
            svc().close_session(session_id)
            return {"ok": True}
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}/pending-permission")
    def pending_permission(session_id: str):
        try:
            return {"pending_permission": svc().pending_permission(session_id)}
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/messages")
    def run_message(session_id: str, payload: MessageRequest):
        try:
            svc().get_session(session_id)
            content = payload.content
            if not content.strip():
                raise InvalidRequestError("message content is required")
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InvalidRequestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return StreamingResponse(
            svc().run_message_sse(session_id, content),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/sessions/{session_id}/permissions/{request_id}")
    def resolve_permission(
        session_id: str,
        request_id: str,
        payload: PermissionDecisionRequest,
    ):
        try:
            svc()._get_pending_request(session_id, request_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionControllerUnavailable as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PermissionRequestNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        return StreamingResponse(
            svc().resolve_permission_sse(session_id, request_id, payload.approved),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app() if FastAPI is not None else None
