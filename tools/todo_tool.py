"""Tool wrapper for TodoList updates."""

from __future__ import annotations

from typing import Optional

from core.todo import TodoList
from core.todo_store import TodoStore
from tools.base_tool import BaseTool, SAFE


class TodoUpdateTool(BaseTool):
    name = "update_todo"
    risk_level = SAFE
    description = (
        "更新当前任务列表。用于复杂任务时创建计划、标记进行中/完成/阻塞，"
        "让用户和 Agent 都能看到当前进度。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "操作类型：add/start/done/complete/block/cancel/reset/remove/clear/show。",
            },
            "description": {"type": "string", "description": "add 时的任务描述。"},
            "index": {"type": "integer", "description": "要操作的任务序号，从 1 开始。"},
            "result": {"type": "string", "description": "完成结果或阻塞/取消原因。"},
        },
        "required": ["action"],
    }

    def __init__(self, todo_list: TodoList, store: Optional[TodoStore] = None):
        self.todo_list = todo_list
        self.store = store

    def execute(
        self,
        action: str,
        description: Optional[str] = None,
        index: Optional[int] = None,
        result: Optional[str] = None,
        **kwargs,
    ) -> str:
        action = action.lower().strip()
        changed = False

        if action == "add":
            if not description:
                return "[错误] add 操作需要 description"
            task = self.todo_list.add(description)
            changed = True
            message = f"[成功] 已添加任务: {task.description}"

        elif action == "start":
            if index is None:
                return "[错误] start 操作需要 index"
            ok = self.todo_list.start(index)
            changed = ok
            message = f"[成功] 任务 {index} 已标记进行中" if ok else f"[错误] 无效任务序号: {index}"

        elif action in {"done", "complete"}:
            if index is None:
                return "[错误] done 操作需要 index"
            ok = self.todo_list.complete(index, result)
            changed = ok
            message = f"[成功] 任务 {index} 已完成" if ok else f"[错误] 无效任务序号: {index}"

        elif action == "block":
            if index is None:
                return "[错误] block 操作需要 index"
            ok = self.todo_list.block(index, result)
            changed = ok
            message = f"[成功] 任务 {index} 已阻塞" if ok else f"[错误] 无效任务序号: {index}"

        elif action == "cancel":
            if index is None:
                return "[错误] cancel 操作需要 index"
            ok = self.todo_list.cancel(index, result)
            changed = ok
            message = f"[成功] 任务 {index} 已取消" if ok else f"[错误] 无效任务序号: {index}"

        elif action == "reset":
            if index is None:
                return "[错误] reset 操作需要 index"
            ok = self.todo_list.reset(index)
            changed = ok
            message = f"[成功] 任务 {index} 已重置" if ok else f"[错误] 无效任务序号: {index}"

        elif action == "remove":
            if index is None:
                return "[错误] remove 操作需要 index"
            removed = self.todo_list.remove(index)
            changed = removed is not None
            message = f"[成功] 已移除任务: {removed.description}" if removed else f"[错误] 无效任务序号: {index}"

        elif action == "clear":
            self.todo_list.clear()
            changed = True
            message = "[成功] TodoList 已清空"

        elif action == "show":
            return self.todo_list.format_for_prompt()

        else:
            return f"[错误] 未知 action: {action}"

        if changed and self.store:
            self.store.save(self.todo_list)

        return message + "\n" + self.todo_list.format_for_prompt()
