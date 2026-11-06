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
from tools.security import (
    contains_sudo,
    contains_sensitive_reference,
    contains_system_path_reference,
    is_inside_workspace,
    is_system_path,
    parse_read_only_shell_command,
    path_security_reason,
    resolve_path,
)


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
        self.project_root = os.path.realpath(os.path.abspath(project_root or os.getcwd()))
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.require_approval = require_approval
        self.allow_dangerous = allow_dangerous
        self.enforce_workspace = enforce_workspace
        self.last_denial_reason = ""

    def approve(self, tool_name: str, args: dict) -> bool:
        self.last_denial_reason = ""
        decision = self.decide(tool_name, args)
        if not decision.allowed:
            self.last_denial_reason = decision.reason
            self.output_fn(self._format_block_message(decision))
            return False

        if not decision.requires_approval:
            return True

        approved = self.request_approval(decision, args)
        if not approved:
            self.last_denial_reason = "用户拒绝了本次操作，请停止重试。"
        return approved

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

        code_decision = self._check_code_access(tool_name, args)
        if code_decision is not None:
            return code_decision

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
            if parse_read_only_shell_command(args.get("code", "")) is not None:
                return SAFE, "受限只读命令，可自动执行"
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
        # Browsing, editing, and writing all pass through this check.  A
        # sensitive target is denied before an approval prompt so a caller
        # cannot accidentally approve a secret read or overwrite.
        sensitive_reason = path_security_reason(
            path,
            self.project_root,
            check_readability=tool_name not in WORKSPACE_WRITE_TOOLS,
        )
        if sensitive_reason is not None:
            return PermissionDecision(
                tool_name=tool_name,
                risk_level=DANGEROUS,
                allowed=False,
                reason=sensitive_reason,
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

    def _check_shell_command(self, tool_name: str, args: dict) -> Optional[PermissionDecision]:
        if tool_name != "execute_code" or args.get("mode") != "shell":
            return None

        command = args.get("code", "")
        if contains_sudo(command):
            return PermissionDecision(
                tool_name=tool_name,
                risk_level=DANGEROUS,
                allowed=False,
                reason="拒绝执行 sudo 命令；当前工具不会提权，请勿重试。",
            )
        if contains_sensitive_reference(command):
            return PermissionDecision(
                tool_name=tool_name, risk_level=DANGEROUS, allowed=False,
                reason="拒绝执行包含敏感文件访问的命令；请停止重试。",
            )
        if contains_system_path_reference(command):
            return PermissionDecision(
                tool_name=tool_name, risk_level=DANGEROUS, allowed=False,
                reason="拒绝执行包含系统路径访问的命令；当前工具不会提权，请停止重试。",
            )
        parsed = parse_read_only_shell_command(command)
        if parsed is None:
            return None

        outside_path = None
        for path in parsed["paths"]:
            resolved = self._resolve_path(path)
            protected_reason = path_security_reason(path, self.project_root)
            if protected_reason is not None:
                return PermissionDecision(
                    tool_name=tool_name,
                    risk_level=DANGEROUS,
                    allowed=False,
                    reason=protected_reason,
                )
            if self.enforce_workspace and not is_inside_workspace(resolved, self.project_root):
                outside_path = resolved

        if outside_path:
            return PermissionDecision(
                tool_name=tool_name, risk_level=CAUTION, allowed=True,
                reason=f"只读命令访问工作区外路径需要确认: {outside_path}",
                requires_approval=self.require_approval,
            )

        return PermissionDecision(
            tool_name=tool_name,
            risk_level=SAFE,
            allowed=True,
            reason="受限只读命令，可自动执行",
        )

    def _check_code_access(self, tool_name: str, args: dict) -> Optional[PermissionDecision]:
        if tool_name != "execute_code":
            return None
        if args.get("mode") == "shell":
            return self._check_shell_command(tool_name, args)
        code = args.get("code", "")
        if contains_sensitive_reference(code):
            return PermissionDecision(
                tool_name=tool_name, risk_level=DANGEROUS, allowed=False,
                reason="拒绝执行包含敏感文件访问的代码；请停止重试。",
            )
        if contains_system_path_reference(code):
            return PermissionDecision(
                tool_name=tool_name, risk_level=DANGEROUS, allowed=False,
                reason="拒绝执行包含系统路径访问的代码；当前工具不会提权，请停止重试。",
            )
        return None

    def _resolve_path(self, path: str) -> str:
        return resolve_path(path, self.project_root)

    def _is_inside_project(self, path: str) -> bool:
        return is_inside_workspace(path, self.project_root)

    @staticmethod
    def _is_system_path(path: str) -> bool:
        return is_system_path(path)

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
