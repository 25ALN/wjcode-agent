import os
import logging
from typing import Optional
from tools.base_tool import BaseTool, SAFE, CAUTION
from tools.security import is_inside_workspace, path_security_reason, resolve_path

logger = logging.getLogger(__name__)


class FileReadTool(BaseTool):
    """读取本地文件内容

    支持：
    - 完整文件读取
    - 指定行号范围读取（start_line ~ end_line）
    - 自动处理 UTF-8 编码
    """

    name = "read_file"
    risk_level = SAFE
    description = (
        "读取指定路径的文件内容。"
        "可以读取整个文件，也可以通过 start_line/end_line 参数只读取部分行。"
        "适用于查看代码、配置文件、文档等场景。"
    )

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "要读取的文件路径。可以是绝对路径（如 /home/user/file.txt）"
                    "或相对于工作区根目录的路径（如 src/main.py）"
                ),
            },
            "start_line": {
                "type": "integer",
                "description": (
                    "可选。起始行号（从1开始，包含该行）。"
                    "例如 start_line=10 表示从第10行开始读取。"
                ),
            },
            "end_line": {
                "type": "integer",
                "description": (
                    "可选。结束行号（从1开始，包含该行）。"
                    "例如 end_line=20 表示读取到第20行。"
                    "如果不指定且 start_line 也不指定，则默认只读取文件开头窗口。"
                ),
            },
        },
        "required": ["path"],
    }

    DEFAULT_MAX_LINES = 160
    DEFAULT_MAX_CHARS = 6000

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = os.path.abspath(workspace_root or os.getcwd())

    def execute(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        读取文件内容

        Args:
            path: 文件路径
            start_line: 起始行号（1-based，可选）
            end_line: 结束行号（1-based，可选）

        Returns:
            文件内容字符串。如果指定了行号范围，每行带行号前缀。
        """
        # 路径存在性检查
        abs_path = self._resolve_path(path)
        security_reason = path_security_reason(abs_path, self.workspace_root)
        if security_reason:
            return f"[拒绝] {security_reason}"
        if not os.path.exists(abs_path):
            return f"[错误] 文件不存在: {abs_path}"

        if not os.path.isfile(abs_path):
            return f"[错误] 路径不是文件: {abs_path}"

        # 尝试读取
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            # 尝试其他常见编码
            for enc in ["gbk", "latin-1"]:
                try:
                    with open(abs_path, "r", encoding=enc) as f:
                        lines = f.readlines()
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            else:
                return f"[错误] 无法解码文件 '{abs_path}'（尝试了 utf-8, gbk, latin-1）"
        except PermissionError:
            return f"[权限拒绝] 当前进程没有读取权限: {abs_path}。不会提权，请停止重试。"
        except OSError as e:
            return f"[错误] 读取文件失败: {e}"

        total_lines = len(lines)
        default_window = start_line is None and end_line is None

        if total_lines == 0:
            logger.debug(f"FileReadTool: 读取空文件 {abs_path}")
            return f"文件: {abs_path} (共 0 行，文件为空)"

        # 未指定范围时只返回开头窗口，避免长文件撑爆模型上下文。
        if default_window:
            start_line = 1
            end_line = min(total_lines, self.DEFAULT_MAX_LINES)
        else:
            if start_line is None:
                start_line = 1
            if end_line is None:
                end_line = total_lines

        # 边界修正
        start_line = max(1, min(start_line, total_lines))
        end_line = max(start_line, min(end_line, total_lines))

        selected = lines[start_line - 1 : end_line]
        rendered_lines = []
        rendered_chars = 0
        displayed_end = start_line - 1
        truncated_by_chars = False

        for i, line in enumerate(selected, start=start_line):
            rendered = f"{i:4d} | {line.rstrip()}"
            if default_window and rendered_chars + len(rendered) + 1 > self.DEFAULT_MAX_CHARS:
                truncated_by_chars = True
                if not rendered_lines:
                    prefix = f"{i:4d} | "
                    budget = max(0, self.DEFAULT_MAX_CHARS - len(prefix) - 20)
                    rendered_lines.append(prefix + line.rstrip()[:budget] + "...(单行过长已截断)")
                    displayed_end = i
                break
            rendered_lines.append(rendered)
            rendered_chars += len(rendered) + 1
            displayed_end = i

        if displayed_end < start_line:
            displayed_end = start_line

        # 拼接输出，带行号
        result_parts = [f"文件: {abs_path} (共 {total_lines} 行，显示第 {start_line}-{displayed_end} 行)"]
        result_parts.append("-" * 50)
        result_parts.extend(rendered_lines)

        if default_window and (displayed_end < total_lines or truncated_by_chars):
            result_parts.append("-" * 50)
            result_parts.append(
                "[提示] 默认读取已截断："
                f"文件共 {total_lines} 行，本次显示第 {start_line}-{displayed_end} 行。"
                "如需更多内容，请再次调用 read_file 并指定 start_line/end_line。"
            )

        logger.debug(
            f"FileReadTool: 读取 {abs_path} "
            f"(行 {start_line}-{displayed_end}/{total_lines}, {len(rendered_lines)} 行)"
        )
        return "\n".join(result_parts)

    def _resolve_path(self, path: str) -> str:
        """解析路径：如果 path 是相对路径，相对于工作区根目录。"""
        if os.path.isabs(path):
            return os.path.abspath(path)
        return os.path.abspath(os.path.join(self.workspace_root, path))


class ReadFilesTool(BaseTool):
    """Read several workspace files in one project-analysis tool call.

    Each item is checked and rendered independently.  A protected or
    out-of-workspace item therefore cannot block ordinary source files in the
    same batch, and it never becomes an approval opportunity.
    """

    name = "read_files"
    risk_level = SAFE
    MAX_FILES = 8
    MAX_OUTPUT_CHARS = 32_000
    description = (
        "批量读取项目内多个普通文件，仅用于项目分析。"
        "敏感文件、系统路径、工作区外路径和无读取权限的文件会逐项拒绝。"
        "每次最多 8 个文件，总输出最多 32000 字符。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "description": "要读取的文件请求列表，最多 8 项。",
                "maxItems": MAX_FILES,
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对于工作区的文件路径。"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["files"],
        "additionalProperties": False,
    }

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = os.path.abspath(workspace_root or os.getcwd())
        self._single_reader = FileReadTool(workspace_root=self.workspace_root)

    def execute(self, files: list[dict], **kwargs) -> str:
        if not isinstance(files, list) or not files:
            return "[错误] read_files 需要至少一个文件请求。"
        if len(files) > self.MAX_FILES:
            return f"[错误] read_files 一次最多读取 {self.MAX_FILES} 个文件。"

        requests = []
        seen = set()
        for item in files:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                requests.append((None, "[错误] 文件请求必须包含字符串 path。"))
                continue
            path = item["path"]
            start_line = item.get("start_line")
            end_line = item.get("end_line")
            key = (path, start_line, end_line)
            if key in seen:
                continue
            seen.add(key)
            requests.append((
                {"path": path, "start_line": start_line, "end_line": end_line},
                None,
            ))

        if not requests:
            return "[错误] 没有有效的文件请求。"

        parts = [f"[批量读取] 请求 {len(requests)} 项，最多返回 {self.MAX_OUTPUT_CHARS} 字符。"]
        used = len(parts[0]) + 1
        for index, (request, validation_error) in enumerate(requests, 1):
            if request is None:
                content = validation_error
                label = "<invalid>"
            else:
                path = request["path"]
                label = path
                resolved = resolve_path(path, self.workspace_root)
                if not is_inside_workspace(resolved, self.workspace_root):
                    content = (
                        "[拒绝] 项目分析批量读取仅允许工作区内文件: "
                        f"{resolved}。请停止重试。"
                    )
                else:
                    content = self._single_reader.execute(**request)

            section = f"\n=== 文件: {label} ===\n{content}\n"
            remaining = self.MAX_OUTPUT_CHARS - used
            if remaining <= 0:
                parts.append("[提示] 批量输出已达到 32000 字符上限，后续文件未展开。")
                break
            if len(section) > remaining:
                marker = "\n[提示] 批量输出已截断。"
                parts.append(section[:max(0, remaining - len(marker))] + marker[:remaining])
                break
            parts.append(section)
            used += len(section)

        return "".join(parts)


class FileWriteTool(BaseTool):
    """将内容写入本地文件
    支持：
    - 创建新文件
    - 覆盖已有文件
    - 追加模式（append=True）
    - 自动创建父目录
    """

    name = "write_file"
    risk_level = CAUTION
    description = (
        "将指定内容写入文件。默认覆盖模式（会替换文件全部内容），"
        "可以通过 append=True 切换为追加模式（在文件末尾添加内容）。"
        "如果父目录不存在会自动创建。"
        "适用于生成代码文件、保存结果、创建配置等场景。"
    )

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "要写入的文件路径。可以是绝对路径或相对于工作区根目录的路径。"
                    "如果父目录不存在会自动创建。"
                ),
            },
            "content": {
                "type": "string",
                "description": "要写入文件的文本内容。",
            },
            "append": {
                "type": "boolean",
                "description": (
                    "是否以追加模式写入。"
                    "False（默认）：覆盖整个文件。"
                    "True：在文件末尾追加内容。"
                ),
            },
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = os.path.abspath(workspace_root or os.getcwd())

    def execute(
        self,
        path: str,
        content: str,
        append: bool = False,
        **kwargs,
    ) -> str:
        """
        Args:
            path: 文件路径
            content: 要写入的内容
            append: 是否追加模式（默认覆盖）

        Returns:
            操作结果描述
        """
        abs_path = self._resolve_path(path)
        security_reason = path_security_reason(
            abs_path,
            self.workspace_root,
            check_readability=False,
        )
        if security_reason:
            return f"[拒绝] {security_reason}"

        # 安全检查：不允许写入到 /dev, /proc, /sys 等系统目录
        if self._is_restricted_path(abs_path):
            return f"[错误] 禁止写入系统路径: {abs_path}"

        # 确保父目录存在
        parent_dir = os.path.dirname(abs_path)
        if parent_dir and not os.path.exists(parent_dir):
            try:
                os.makedirs(parent_dir, exist_ok=True)
            except OSError as e:
                return f"[错误] 无法创建目录 '{parent_dir}': {e}"

        # 写入文件
        mode = "a" if append else "w"
        action = "追加" if append else "覆盖写入"

        try:
            with open(abs_path, mode, encoding="utf-8") as f:
                f.write(content)
            file_size = os.path.getsize(abs_path)
            logger.debug(
                f"FileWriteTool: {action} {abs_path} ({file_size} 字节)"
            )
            return (
                f"[成功] {action}文件: {abs_path}\n"
                f"写入 {len(content)} 字符，文件大小 {file_size} 字节"
            )
        except PermissionError:
            return f"[错误] 没有权限写入文件: {abs_path}"
        except OSError as e:
            return f"[错误] 写入文件失败: {e}"

    def _resolve_path(self, path: str) -> str:
        """解析路径：如果 path 是相对路径，相对于工作区根目录。"""
        if os.path.isabs(path):
            return os.path.abspath(path)
        return os.path.abspath(os.path.join(self.workspace_root, path))

    @staticmethod
    def _is_restricted_path(path: str) -> bool:
        """检查是否为受限系统路径"""
        restricted_prefixes = [
            "/dev/",
            "/proc/",
            "/sys/",
            "/boot/",
        ]
        normalized = os.path.normpath(path)
        for prefix in restricted_prefixes:
            if normalized.startswith(prefix):
                return True
        return False
