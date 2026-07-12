"""Permission system for tool execution.

The manager keeps safe tools automatic, asks for explicit approval for risky
operations, and blocks dangerous actions when configured to do so.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from tools.base_tool import SAFE, CAUTION, DANGEROUS
from tools.registry import ToolRegistry


RISK_ORDER = {
    SAFE: 0,
    CAUTION: 1,
    DANGEROUS: 2,
}

PATH_AWARE_TOOLS = {"read_file", "write_file", "edit_file", "ls", "grep"}
WORKSPACE_WRITE_TOOLS = {"write_file", "edit_file"}


@dataclass
class PermissionDecision:
    tool_name: str
    risk_level: str
    allowed: bool
    reason: str
    requires_approval: bool = False


class PermissionManager:
    """Classifies tool risk and requests approval before execution."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        project_root: Optional[str] = None,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        require_approval: bool = True,
        allow_dangerous: bool = True,
        enforce_workspace: bool = True,
    ):
        self.tool_registry = tool_registry
        self.project_root = os.path.abspath(project_root or os.getcwd())
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.require_approval = require_approval
        self.allow_dangerous = allow_dangerous
        self.enforce_workspace = enforce_workspace

    def approve(self, tool_name: str, args: dict) -> bool:
        decision = self.decide(tool_name, args)
        if not decision.allowed:
            self.output_fn(self._format_block_message(decision))
            return False

        if not decision.requires_approval:
            return True

        return self.request_approval(decision, args)

    def decide(self, tool_name: str, args: dict) -> PermissionDecision:
        tool = self.tool_registry.get(tool_name) if self.tool_registry else None
        if tool is None:
            return PermissionDecision(
                tool_name=tool_name,
                risk_level=DANGEROUS,
                allowed=False,
                reason="未知工具",
            )

        risk_level = getattr(tool, "risk_level", SAFE)
        reasons = []

        dynamic_risk, dynamic_reason = self._dynamic_risk(tool_name, args)
        risk_level = self._max_risk(risk_level, dynamic_risk)
        if dynamic_reason:
            reasons.append(dynamic_reason)

        path_decision = self._check_paths(tool_name, args)
        if path_decision is not None:
            return path_decision

        if risk_level == SAFE:
            return PermissionDecision(
                tool_name=tool_name,
                risk_level=risk_level,
                allowed=True,
                reason="安全工具，可自动执行",
            )

        if risk_level == DANGEROUS and not self.allow_dangerous:
            return PermissionDecision(
                tool_name=tool_name,
                risk_level=risk_level,
                allowed=False,
                reason="危险工具被策略禁止",
            )

        return PermissionDecision(
            tool_name=tool_name,
            risk_level=risk_level,
            allowed=True,
            reason="；".join(reasons) if reasons else "工具需要用户确认",
            requires_approval=self.require_approval,
        )

    def request_approval(self, decision: PermissionDecision, args: dict) -> bool:
        self.output_fn("")
        self.output_fn("⚠️  工具执行需要确认")
        self.output_fn(f"工具: {decision.tool_name}")
        self.output_fn(f"风险等级: {decision.risk_level}")
        self.output_fn(f"原因: {decision.reason}")
        self.output_fn(f"参数: {self._format_args(args)}")
        answer = self.input_fn("是否允许执行？输入 y/yes 继续: ").strip().lower()
        return answer in {"y", "yes"}

    def _dynamic_risk(self, tool_name: str, args: dict) -> tuple[str, str]:
        if tool_name == "execute_code" and args.get("mode") == "shell":
            return DANGEROUS, "shell 命令可执行任意系统操作"

        if tool_name == "web_search":
            return CAUTION, "联网搜索可能泄露查询内容"

        return SAFE, ""

    def _check_paths(self, tool_name: str, args: dict) -> Optional[PermissionDecision]:
        if tool_name not in PATH_AWARE_TOOLS:
            return None

        path = args.get("path")
        if path is None and tool_name == "ls":
            path = "."
        if path is None and tool_name == "grep":
            path = "."
        if not isinstance(path, str):
            return None

        abs_path = self._resolve_path(path)
        if self._is_system_path(abs_path):
            return PermissionDecision(
                tool_name=tool_name,
                risk_level=DANGEROUS,
                allowed=False,
                reason=f"拒绝访问系统路径: {abs_path}",
            )

        if self.enforce_workspace and not self._is_inside_project(abs_path):
            risk_level = DANGEROUS if tool_name in WORKSPACE_WRITE_TOOLS else CAUTION
            if risk_level == DANGEROUS and not self.allow_dangerous:
                return PermissionDecision(
                    tool_name=tool_name,
                    risk_level=risk_level,
                    allowed=False,
                    reason=f"拒绝访问工作区外路径: {abs_path}",
                )
            return PermissionDecision(
                tool_name=tool_name,
                risk_level=risk_level,
                allowed=True,
                reason=f"访问工作区外路径需要确认: {abs_path}",
                requires_approval=self.require_approval,
            )

        return None

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return os.path.abspath(path)
        return os.path.abspath(os.path.join(self.project_root, path))

    def _is_inside_project(self, path: str) -> bool:
        try:
            return os.path.commonpath([self.project_root, os.path.abspath(path)]) == self.project_root
        except ValueError:
            return False

    @staticmethod
    def _is_system_path(path: str) -> bool:
        normalized = os.path.normpath(path)
        restricted_prefixes = (
            "/dev",
            "/proc",
            "/sys",
            "/boot",
            "/etc",
            "/usr/bin",
            "/usr/sbin",
            "/bin",
            "/sbin",
        )
        return normalized in restricted_prefixes or any(
            normalized.startswith(prefix + os.sep) for prefix in restricted_prefixes
        )

    @staticmethod
    def _max_risk(left: str, right: str) -> str:
        return left if RISK_ORDER.get(left, 0) >= RISK_ORDER.get(right, 0) else right

    @staticmethod
    def _format_args(args: dict, max_length: int = 800) -> str:
        text = repr(args)
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    @staticmethod
    def _format_block_message(decision: PermissionDecision) -> str:
        return (
            f"[权限拒绝] Tool '{decision.tool_name}' 未执行。"
            f"风险等级: {decision.risk_level}。原因: {decision.reason}。"
        )
