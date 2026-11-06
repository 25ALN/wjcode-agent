import subprocess
import tempfile
import os
import logging
from tools.base_tool import BaseTool, CAUTION, DANGEROUS
from tools.project_tools import GrepTool, LSTool
from tools.security import (
    contains_sudo,
    contains_sensitive_reference,
    contains_system_path_reference,
    parse_read_only_shell_command,
    path_security_reason,
)

logger = logging.getLogger(__name__)


class CodeExecutorTool(BaseTool):

    name = "execute_code"
    risk_level = CAUTION  # Python 模式为 caution，shell 模式在执行时动态升级为 dangerous
    description = (
        "执行代码并返回结果。"
        "支持两种模式："
        "1) mode='python'（默认）：执行 Python 代码"
        "2) mode='shell'：执行任意 Shell 命令（如 ls、cat、pip install 等）"
        "适用于：运行 AI 生成的代码验证正确性、计算数学题、"
        "查看系统信息、操作文件等。"
        "简单 ls/grep 会由项目浏览工具执行，输出经过过滤且最多返回 1000 项；其他命令需要审批。"
        "注意：有 30 秒超时限制。"
    )

    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "description": "执行模式。'python' 表示 Python 代码，'shell' 表示 Shell 命令。默认 'python'。",
            },
            "code": {
                "type": "string",
                "description": (
                    "要执行的代码。如果是 python 模式，写 Python 语句；"
                    "如果是 shell 模式，写 Shell 命令（如 ls -la、cat file.txt）"
                ),
            },
        },
        "required": ["code"],
    }

    def __init__(self, workspace_root: str | None = None):
        self.workspace_root = os.path.abspath(workspace_root or os.getcwd())

    def execute(self, code: str, mode: str = "python", **kwargs) -> str:
        if mode == "shell":
            return self._run_shell(code)
        return self._run_python(code)

    def _run_python(self, code: str) -> str:
        if contains_sensitive_reference(code):
            return "[拒绝] 代码包含敏感文件访问；当前工具不会读取受保护文件，请停止重试。"
        if contains_system_path_reference(code):
            return "[拒绝] 代码包含系统路径访问；当前工具不会提权，请停止重试。"
        tmp_path = None
        try:
            tmp_path = self._write_temp_file(code, suffix=".py")
            result = subprocess.run(
                ["python3", tmp_path],
                capture_output=True, text=True, timeout=30,
                cwd=self.workspace_root,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            return self._format_output("python", result)
        except subprocess.TimeoutExpired:
            return "[错误] 代码执行超时（30秒），请检查是否有死循环。"
        except FileNotFoundError:
            return "[错误] 未找到 python3 解释器。"
        except Exception as e:
            return f"[错误] 执行异常: {str(e)[:500]}"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _run_shell(self, command: str) -> str:
        if contains_sudo(command):
            return "[拒绝] 不执行 sudo 命令；当前工具不会提权，请勿重试。"
        if contains_sensitive_reference(command):
            return "[拒绝] 命令包含敏感文件访问；当前工具不会读取受保护文件，请停止重试。"
        if contains_system_path_reference(command):
            return "[拒绝] 命令包含系统路径访问；当前工具不会提权，请停止重试。"
        # Auto-approved read-only commands are executed through constrained
        # tool implementations instead of a raw shell, so grep cannot walk
        # into .env, key files, or unreadable files.
        parsed = parse_read_only_shell_command(command)
        if parsed is not None:
            return self._run_safe_read_only(parsed)

        # 危险命令黑名单
        dangerous = ["rm -rf /", "mkfs.", "dd if=", ":(){ :|:& };:", "> /dev/sda"]
        lower = command.lower()
        for d in dangerous:
            if d in lower:
                return f"[拒绝] 检测到危险命令: '{d}'。该命令已被拦截。"

        try:
            result = subprocess.run(
                command, shell=True,
                capture_output=True, text=True, timeout=30,
                cwd=self.workspace_root,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            return self._format_output("shell", result, command)
        except subprocess.TimeoutExpired:
            return f"[错误] Shell 命令超时（30秒）: {command}"
        except Exception as e:
            return f"[错误] Shell 执行异常: {str(e)[:500]}"

    def _run_safe_read_only(self, parsed: dict) -> str:
        paths = parsed.get("paths") or ["."]
        for path in paths:
            reason = path_security_reason(path, self.workspace_root)
            if reason:
                return f"[拒绝] {reason}"

        if parsed.get("name") == "grep":
            tool = GrepTool(workspace_root=self.workspace_root)
            results = []
            for path in paths:
                result = tool.execute(
                    pattern=parsed["pattern"],
                    path=path,
                    max_results=1000,
                )
                results.append(result)
            return "\n".join(results)

        # Route ``ls`` through the project tool so protected names and symlinked
        # entries are filtered instead of being exposed by the system binary.
        tool = LSTool(workspace_root=self.workspace_root)
        results = []
        for path in paths:
            results.append(tool.execute(path=path))
        return "\n".join(results)

    @staticmethod
    def _format_output(mode: str, result: subprocess.CompletedProcess, cmd: str = "") -> str:
        parts = []
        if result.stdout:
            parts.append(f"[stdout]\n{result.stdout.rstrip()}")
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr.rstrip()}")
        if not parts:
            parts.append("[执行完成，无输出]")
        parts.append(f"[退出码: {result.returncode}]")
        return "\n".join(parts)

    @staticmethod
    def _write_temp_file(code: str, suffix: str = ".py") -> str:
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="agent_exec_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        return path
