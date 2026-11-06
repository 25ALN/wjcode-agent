"""Shared path and command safety helpers for local tools.

These helpers deliberately live in ``tools`` so both the permission layer and
the tools themselves enforce the same boundaries.  Permission checks alone are
not enough: tools can also be called directly in tests or by another runtime.
"""

from __future__ import annotations

import os
import re
import shlex
import stat
from typing import Any, Optional


SYSTEM_PATH_PREFIXES = (
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

SENSITIVE_DIRECTORY_NAMES = {
    ".aws",
    ".gnupg",
    ".ssh",
}

SENSITIVE_EXACT_NAMES = {
    "credentials",
    "credentials.json",
    "secret",
    "secrets",
    "secrets.json",
    "password",
    "passwords",
    "passwd",
    "id_rsa",
    "id_ed25519",
}

SENSITIVE_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".jks")
SENSITIVE_TOKEN_RE = re.compile(
    r"(?:^|[._-])(secret|credential|password|passwd|api[_-]?key|access[_-]?key)(?:[._-]|$)",
    re.IGNORECASE,
)


def resolve_path(path: str, workspace_root: Optional[str] = None) -> str:
    """Resolve a user path and follow symlinks for security checks."""

    root = os.path.realpath(os.path.abspath(workspace_root or os.getcwd()))
    raw = path if os.path.isabs(path) else os.path.join(root, path)
    return os.path.realpath(os.path.abspath(raw))


def is_system_path(path: str) -> bool:
    normalized = os.path.normpath(os.path.realpath(path))
    return normalized in SYSTEM_PATH_PREFIXES or any(
        normalized.startswith(prefix + os.sep) for prefix in SYSTEM_PATH_PREFIXES
    )


def is_sensitive_name(name: str) -> bool:
    """Return whether a basename should never be read by browsing tools."""

    lowered = str(name).lower()
    if lowered.startswith(".env"):
        return True
    if lowered in SENSITIVE_EXACT_NAMES or lowered.endswith(SENSITIVE_SUFFIXES):
        return True
    if lowered.startswith(("id_rsa", "id_ed25519")):
        return True
    return SENSITIVE_TOKEN_RE.search(lowered) is not None


def is_sensitive_path(path: str) -> bool:
    """Check the real path, including sensitive parent directories."""

    normalized = os.path.normpath(os.path.realpath(path))
    return any(
        is_sensitive_name(part)
        or part.lower() in SENSITIVE_DIRECTORY_NAMES
        for part in normalized.split(os.sep)
        if part
    )


def path_security_reason(
    path: str,
    workspace_root: Optional[str] = None,
    *,
    check_readability: bool = True,
) -> Optional[str]:
    """Return a user-facing denial reason, or ``None`` when the path is safe.

    Write operations can disable readability checks. Browsing tools check
    both their starting path and every entry before opening it.
    """

    lexical = os.path.abspath(os.path.join(workspace_root or os.getcwd(), path))
    resolved = resolve_path(path, workspace_root)
    if is_system_path(resolved):
        return (
            f"拒绝访问系统路径: {resolved}。"
            "当前工具不会提权，请勿重试或使用 sudo。"
        )
    if any(is_sensitive_name(part) or part.lower() in SENSITIVE_DIRECTORY_NAMES
           for part in lexical.split(os.sep)) or is_sensitive_path(resolved):
        return (
            f"拒绝读取敏感路径: {resolved}。"
            "该路径受保护，请勿重试或使用 sudo。"
        )
    if check_readability:
        try:
            mode = os.stat(resolved).st_mode
            has_read_bit = bool(mode & (stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH))
            can_read = os.access(resolved, os.R_OK)
            can_traverse = not stat.S_ISDIR(mode) or os.access(resolved, os.X_OK)
            if not has_read_bit or not can_read or not can_traverse:
                return (
                    f"当前进程没有读取权限: {resolved}。"
                    "不会尝试提权，请停止重试。"
                )
        except FileNotFoundError:
            pass
        except OSError as exc:
            return f"无法检查路径权限: {resolved} ({exc})。请停止重试。"
    return None


def is_inside_workspace(path: str, workspace_root: str) -> bool:
    try:
        return os.path.commonpath([
            os.path.realpath(os.path.abspath(workspace_root)),
            resolve_path(path, workspace_root),
        ]) == os.path.realpath(os.path.abspath(workspace_root))
    except ValueError:
        return False


def contains_sudo(command: str) -> bool:
    try:
        tokens = shlex.split(command, posix=True)
    except (TypeError, ValueError):
        tokens = str(command or "").split()
    return any(os.path.basename(token) == "sudo" for token in tokens)


def contains_sensitive_reference(command: str) -> bool:
    """Conservatively detect protected file names in code or commands."""

    if not isinstance(command, str):
        return False
    try:
        tokens = shlex.split(command, posix=True)
    except (TypeError, ValueError):
        tokens = str(command or "").split()
    for token in tokens:
        cleaned = token.strip("\"'()[]{}:,;")
        if is_sensitive_name(os.path.basename(cleaned)):
            return True
        if any(part.lower() in SENSITIVE_DIRECTORY_NAMES for part in cleaned.replace("\\", "/").split("/")):
            return True
    return bool(re.search(
        r"(?:^|[/\\.'\"\s(=])(?:\.env(?:[.\w-]*)?|\.aws|\.ssh|\.gnupg|"
        r"id_(?:rsa|ed25519)(?:[.\w-]*)?)(?=$|[/\\.'\"\s),;])",
        command,
        re.IGNORECASE,
    ))


def contains_system_path_reference(command: str) -> bool:
    """Detect explicit references to protected system paths."""

    if not isinstance(command, str):
        return False
    for match in re.findall(
        r"(?<![A-Za-z0-9_.-])/(?:dev|proc|sys|boot|etc|usr/bin|usr/sbin|bin|sbin)"
        r"(?:[/\\][^\s'\"`;|&><)]*)?",
        command,
    ):
        if is_system_path(match):
            return True
    return False


def parse_read_only_shell_command(command: str) -> Optional[dict[str, Any]]:
    """Parse the deliberately small shell auto-approval allowlist.

    The parser rejects shell operators and options that can introduce another
    input file or recursive behavior we cannot constrain.  ``grep -R`` is safe
    because the executor routes it through the project GrepTool, which filters
    protected and unreadable files.
    """

    if not isinstance(command, str) or not command.strip() or contains_sudo(command):
        return None
    if any(char in command for char in "\n\r;|&><`$*?[]{}~"):
        return None
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return None
    if not tokens:
        return None

    executable = os.path.basename(tokens[0])
    if executable == "ls":
        paths: list[str] = []
        for token in tokens[1:]:
            if token == "--":
                continue
            if token.startswith("-"):
                # Listing is filtered by LSTool even with hidden-file flags.
                if token in {"--all", "--almost-all"} or (not token.startswith("--") and set(token[1:]) <= set("laA1F")):
                    continue
                return None
            paths.append(token)
        return {"name": "ls", "tokens": tokens, "paths": paths or ["."]}

    if executable != "grep":
        return None

    index = 1
    recursive = False
    while index < len(tokens) and tokens[index].startswith("-") and tokens[index] != "-":
        option = tokens[index]
        if option == "--recursive":
            recursive = True
        elif option == "--line-number":
            pass
        elif not option.startswith("--") and set(option[1:]) <= set("rRn"):
            recursive = recursive or "r" in option or "R" in option
        else:
            return None
        index += 1
    if index >= len(tokens):
        return None
    pattern = tokens[index]
    paths = tokens[index + 1:]
    if not paths or any(path.startswith("-") for path in paths):
        return None  # Never read stdin or wait for a pipe in an auto-approved call.
    return {
        "name": "grep",
        "tokens": tokens,
        "pattern": pattern,
        "paths": paths,
        "recursive": recursive,
    }

