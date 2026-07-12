"""Project navigation and editing tools."""

from __future__ import annotations

import os
import re
import difflib
from fnmatch import fnmatch
from typing import Iterable, List

from tools.base_tool import BaseTool, SAFE, CAUTION


DEFAULT_EXCLUDES = {
    ".git",
    ".agents",
    ".codex",
    "venv",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
}


class LSTool(BaseTool):
    name = "ls"
    risk_level = SAFE
    description = "列出项目目录结构，自动过滤 venv、__pycache__、.git 等噪音目录。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要列出的目录，默认工作区根目录。"},
            "recursive": {"type": "boolean", "description": "是否递归列出子目录，默认 false。"},
            "max_entries": {"type": "integer", "description": "最多返回条目数，默认 200。"},
        },
        "required": [],
    }

    def __init__(self, workspace_root: str | None = None):
        self.workspace_root = os.path.abspath(workspace_root or os.getcwd())

    def execute(self, path: str = ".", recursive: bool = False, max_entries: int = 200, **kwargs) -> str:
        root = _resolve_path(path, self.workspace_root)
        if not os.path.exists(root):
            return f"[错误] 路径不存在: {root}"
        if not os.path.isdir(root):
            return f"[错误] 路径不是目录: {root}"

        max_entries = _clamp(max_entries, 1, 1000)
        lines = [f"目录: {root}"]
        count = 0

        if recursive:
            for current, dirs, files in os.walk(root):
                dirs[:] = sorted(d for d in dirs if d not in DEFAULT_EXCLUDES and not d.startswith("."))
                files = sorted(files)
                rel = os.path.relpath(current, root)
                depth = 0 if rel == "." else rel.count(os.sep) + 1
                if rel != ".":
                    lines.append(f"{'  ' * (depth - 1)}{os.path.basename(current)}/")
                    count += 1
                for fname in files:
                    if _is_noise_file(fname):
                        continue
                    lines.append(f"{'  ' * depth}{fname}")
                    count += 1
                    if count >= max_entries:
                        lines.append(f"...(已截断，仅显示前 {max_entries} 项)")
                        return "\n".join(lines)
        else:
            for name in sorted(os.listdir(root)):
                if name in DEFAULT_EXCLUDES or _is_noise_file(name):
                    continue
                full = os.path.join(root, name)
                suffix = "/" if os.path.isdir(full) else ""
                lines.append(f"{name}{suffix}")
                count += 1
                if count >= max_entries:
                    lines.append(f"...(已截断，仅显示前 {max_entries} 项)")
                    break

        lines.append(f"共 {count} 项")
        return "\n".join(lines)


class GrepTool(BaseTool):
    name = "grep"
    risk_level = SAFE
    description = "在项目文件中按正则搜索内容，适合查找代码符号、函数名、配置项。"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "正则表达式。"},
            "path": {"type": "string", "description": "搜索目录或文件，默认当前目录。"},
            "file_pattern": {"type": "string", "description": "文件名 glob，如 *.py。"},
            "max_results": {"type": "integer", "description": "最多返回匹配条数，默认 100。"},
        },
        "required": ["pattern"],
    }

    def __init__(self, workspace_root: str | None = None):
        self.workspace_root = os.path.abspath(workspace_root or os.getcwd())

    def execute(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str = "*",
        max_results: int = 100,
        **kwargs,
    ) -> str:
        root = _resolve_path(path, self.workspace_root)
        if not os.path.exists(root):
            return f"[错误] 路径不存在: {root}"

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"[错误] 正则表达式无效: {exc}"

        max_results = _clamp(max_results, 1, 1000)
        files = [root] if os.path.isfile(root) else _iter_files(root, file_pattern)
        matches = []

        for fpath in files:
            if len(matches) >= max_results:
                break
            if os.path.isfile(fpath) and not fnmatch(os.path.basename(fpath), file_pattern):
                continue
            if _looks_binary(fpath):
                continue
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for lineno, line in enumerate(f, 1):
                        if regex.search(line):
                            rel = os.path.relpath(fpath, os.getcwd())
                            matches.append(f"{rel}:{lineno}: {line.rstrip()}")
                            if len(matches) >= max_results:
                                break
            except (UnicodeDecodeError, OSError):
                continue

        header = f"搜索: /{pattern}/ in {root} ({file_pattern})"
        if not matches:
            return header + "\n未找到匹配结果"
        suffix = "" if len(matches) < max_results else f"\n...(已截断，仅显示前 {max_results} 条)"
        return header + "\n" + "\n".join(matches) + suffix


class EditTool(BaseTool):
    name = "edit_file"
    risk_level = CAUTION
    description = "精确编辑文件的指定行范围。比 write_file 全量覆盖更适合小范围代码修改。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "要编辑的文件路径。"},
            "start_line": {"type": "integer", "description": "起始行号，从 1 开始。"},
            "end_line": {"type": "integer", "description": "结束行号，从 1 开始，包含该行。"},
            "new_content": {"type": "string", "description": "替换到指定行范围的新内容。"},
            "dry_run": {"type": "boolean", "description": "仅预览 diff，不实际写入文件。"},
        },
        "required": ["path", "start_line", "end_line", "new_content"],
    }

    def __init__(self, workspace_root: str | None = None):
        self.workspace_root = os.path.abspath(workspace_root or os.getcwd())

    def execute(
        self,
        path: str,
        start_line: int,
        end_line: int,
        new_content: str,
        dry_run: bool = False,
        **kwargs,
    ) -> str:
        abs_path = _resolve_path(path, self.workspace_root)
        if not os.path.exists(abs_path):
            return f"[错误] 文件不存在: {abs_path}"
        if not os.path.isfile(abs_path):
            return f"[错误] 路径不是文件: {abs_path}"
        if start_line < 1 or end_line < start_line:
            return "[错误] 行号范围无效，要求 1 <= start_line <= end_line"

        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            return f"[错误] 文件不是 UTF-8 文本: {abs_path}"
        except OSError as exc:
            return f"[错误] 读取文件失败: {exc}"

        if end_line > len(lines):
            return f"[错误] end_line 超出文件行数: {end_line} > {len(lines)}"

        replacement = new_content.splitlines(keepends=True)
        if new_content and not new_content.endswith("\n"):
            replacement[-1] = replacement[-1] + "\n"

        old_count = end_line - start_line + 1
        original_lines = list(lines)
        lines[start_line - 1:end_line] = replacement

        diff = "\n".join(difflib.unified_diff(
            [line.rstrip("\n") for line in original_lines],
            [line.rstrip("\n") for line in lines],
            fromfile=f"{abs_path}:before",
            tofile=f"{abs_path}:after",
            lineterm="",
        ))
        if diff and len(diff) > 4000:
            diff = diff[:4000] + "\n...(diff 已截断)"

        if dry_run:
            return (
                f"[预览] 将编辑文件: {abs_path}\n"
                f"替换第 {start_line}-{end_line} 行（原 {old_count} 行，新 {len(replacement)} 行）\n"
                f"{diff}"
            )

        try:
            with open(abs_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        except OSError as exc:
            return f"[错误] 写入文件失败: {exc}"

        return (
            f"[成功] 已编辑文件: {abs_path}\n"
            f"替换第 {start_line}-{end_line} 行（原 {old_count} 行，新 {len(replacement)} 行）\n"
            f"{diff}"
        )


def _resolve_path(path: str, workspace_root: str | None = None) -> str:
    if os.path.isabs(path):
        return os.path.abspath(path)
    root = os.path.abspath(workspace_root or os.getcwd())
    return os.path.abspath(os.path.join(root, path))


def _clamp(value: int, low: int, high: int) -> int:
    try:
        return min(max(int(value), low), high)
    except (TypeError, ValueError):
        return low


def _iter_files(root: str, file_pattern: str) -> Iterable[str]:
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in DEFAULT_EXCLUDES and not d.startswith("."))
        for fname in sorted(files):
            if _is_noise_file(fname) or not fnmatch(fname, file_pattern):
                continue
            yield os.path.join(current, fname)


def _is_noise_file(name: str) -> bool:
    return name.endswith((".pyc", ".pyo", ".so", ".dll", ".dylib")) or name in {".DS_Store"}


def _looks_binary(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
        return b"\0" in chunk
    except OSError:
        return True
