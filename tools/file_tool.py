import os
import logging
from typing import Optional
from tools.base_tool import BaseTool, SAFE, CAUTION

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
                    "如果不指定则读取到文件末尾。"
                ),
            },
        },
        "required": ["path"],
    }

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
            return f"[错误] 没有权限读取文件: {abs_path}"
        except OSError as e:
            return f"[错误] 读取文件失败: {e}"

        total_lines = len(lines)

        # 确定行号范围
        if start_line is None:
            start_line = 1
        if end_line is None:
            end_line = total_lines

        # 边界修正
        start_line = max(1, min(start_line, total_lines))
        end_line = max(start_line, min(end_line, total_lines))

        selected = lines[start_line - 1 : end_line]

        # 拼接输出，带行号
        result_parts = [f"文件: {abs_path} (共 {total_lines} 行，显示第 {start_line}-{end_line} 行)"]
        result_parts.append("-" * 50)

        for i, line in enumerate(selected, start=start_line):
            result_parts.append(f"{i:4d} | {line.rstrip()}")

        logger.debug(
            f"FileReadTool: 读取 {abs_path} "
            f"(行 {start_line}-{end_line}/{total_lines}, {len(selected)} 行)"
        )
        return "\n".join(result_parts)

    def _resolve_path(self, path: str) -> str:
        """解析路径：如果 path 是相对路径，相对于工作区根目录。"""
        if os.path.isabs(path):
            return os.path.abspath(path)
        return os.path.abspath(os.path.join(self.workspace_root, path))


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