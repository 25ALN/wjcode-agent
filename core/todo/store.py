"""Persistence helper for TodoList."""

from __future__ import annotations

import json
import os
from typing import Optional

from core.todo import TodoList


class TodoStore:
    def __init__(self, path: str = ".agent_todo.json"):
        self.path = os.path.abspath(path)

    def load(self) -> TodoList:
        if not os.path.exists(self.path):
            return TodoList()
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TodoList.from_dict(data)
        except (OSError, json.JSONDecodeError, ValueError, KeyError):
            return TodoList()

    def save(self, todo: TodoList) -> bool:
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(todo.to_dict(), f, ensure_ascii=False, indent=2)
            return True
        except OSError:
            return False

    def clear(self) -> bool:
        try:
            if os.path.exists(self.path):
                os.unlink(self.path)
            return True
        except OSError:
            return False

    @property
    def exists(self) -> bool:
        return os.path.exists(self.path)
