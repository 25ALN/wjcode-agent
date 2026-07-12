"""Intent matching helpers shared by runtime and planning."""

from __future__ import annotations

import re
from typing import Iterable

_ASCII_KEYWORD = re.compile(r"^[a-z0-9_][a-z0-9_.-]*$")
_FILE_REFERENCE = re.compile(
    r"""(^|[\s`'"(（])([./~]?[a-z0-9_./-]+\.(py|md|txt|json|toml|yaml|yml|js|ts|css|html|sh|ini|env)|readme|dockerfile)(?=$|[\s`'"),，。])""",
    re.IGNORECASE,
)


def contains_keyword(text: str, keywords: Iterable[str]) -> bool:
    lower = str(text or "").lower()
    return any(matches_keyword(lower, keyword) for keyword in keywords)


def matches_keyword(lower_text: str, keyword: str) -> bool:
    key = str(keyword or "").strip().lower()
    if not key:
        return False
    if _ASCII_KEYWORD.fullmatch(key):
        return re.search(rf"(?<![a-z0-9_]){re.escape(key)}(?![a-z0-9_])", lower_text) is not None
    return key in lower_text


def looks_like_file_reference(text: str) -> bool:
    return bool(_FILE_REFERENCE.search(str(text or "").lower()))
