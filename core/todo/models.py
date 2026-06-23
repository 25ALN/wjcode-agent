"""
实现 Claude Code 风格的任务拆解与进度追踪。
Agent 接到任务后先拆解为子任务列表，逐项执行并标记完成。

用法：
    todo = TodoList()
    todo.add("分析项目结构")
    todo.add("编写核心代码")
    todo.start(1)
    todo.complete(1, "发现3个模块，结构清晰")
    todo.format_for_prompt()  # -> 格式化后的任务状态文本
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"         # 被阻塞（promote.txt 未定义，但实际需要）
    CANCELLED = "cancelled"     # 已取消（同上）


@dataclass
class Task:

    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None


class TodoList:

    def __init__(self):
        self._tasks: List[Task] = []


    def add(self, description: str) -> Task:
        """添加一个子任务，返回创建的 Task 对象"""
        task = Task(description=description)
        self._tasks.append(task)
        return task

    def insert(self, index: int, description: str) -> Task:
        """在指定位置插入子任务（1-based index）

        插入位置超出范围时自动追加到末尾。
        """
        task = Task(description=description)
        idx = max(0, min(index - 1, len(self._tasks)))
        self._tasks.insert(idx, task)
        return task

    def remove(self, index: int) -> Optional[Task]:
        """移除指定位置的子任务（1-based index），返回被移除的 Task

        如果 index 无效则返回 None。
        """
        if not (1 <= index <= len(self._tasks)):
            return None
        return self._tasks.pop(index - 1)

    def get(self, index: int) -> Optional[Task]:
        """获取指定位置的子任务（1-based index），不存在返回 None"""
        if not (1 <= index <= len(self._tasks)):
            return None
        return self._tasks[index - 1]

    def list_all(self) -> List[Task]:
        """返回所有子任务的副本"""
        return list(self._tasks)

    # 状态变更
    def start(self, index: int) -> bool:

        task = self.get(index)
        if task is None:
            return False
        task.status = TaskStatus.IN_PROGRESS
        return True

    def complete(self, index: int, result: Optional[str] = None) -> bool:

        task = self.get(index)
        if task is None:
            return False
        task.status = TaskStatus.DONE
        if result is not None:
            task.result = result
        return True

    def block(self, index: int, reason: Optional[str] = None) -> bool:
        task = self.get(index)
        if task is None:
            return False
        task.status = TaskStatus.BLOCKED
        if reason is not None:
            task.result = f"[被阻塞] {reason}"
        return True

    def cancel(self, index: int, reason: Optional[str] = None) -> bool:
        task = self.get(index)
        if task is None:
            return False
        task.status = TaskStatus.CANCELLED
        if reason is not None:
            task.result = f"[已取消] {reason}"
        return True

    def reset(self, index: int) -> bool:
        task = self.get(index)
        if task is None:
            return False
        task.status = TaskStatus.PENDING
        task.result = None
        return True

    def add_batch(self, descriptions: List[str]) -> List[Task]:
        """批量添加子任务"""
        return [self.add(d) for d in descriptions]

    def clear(self) -> None:
        """清空所有任务"""
        self._tasks.clear()

    def count(self) -> int:
        """返回子任务总数"""
        return len(self._tasks)

    def count_by_status(self, status: TaskStatus) -> int:
        """返回指定状态的子任务数"""
        return sum(1 for t in self._tasks if t.status == status)

    def is_all_done(self) -> bool:
        """所有子任务是否都已完成"""
        return self.count() > 0 and all(
            t.status == TaskStatus.DONE for t in self._tasks
        )

    def progress(self) -> float:
        """已完成比例（0.0 ~ 1.0）"""
        if self.count() == 0:
            return 0.0
        done = self.count_by_status(TaskStatus.DONE)
        return done / self.count()

    # ── 格式化 ──────────────────────────────

    STATUS_ICONS = {
        TaskStatus.PENDING:    "⬜",
        TaskStatus.IN_PROGRESS: "🔄",
        TaskStatus.DONE:       "✅",
        TaskStatus.BLOCKED:    "🚫",
        TaskStatus.CANCELLED:  "❌",
    }

    def format_for_prompt(self) -> str:

        if self.count() == 0:
            return "（暂无子任务）"

        done = self.count_by_status(TaskStatus.DONE)
        total = self.count()
        lines = [f"任务进度（{done}/{total} 已完成，{self.progress():.0%}）："]

        for i, task in enumerate(self._tasks, 1):
            icon = self.STATUS_ICONS.get(task.status, "❓")
            suffix = ""
            if task.result:
                # 截断过长的结果
                result_text = task.result
                if len(result_text) > 80:
                    result_text = result_text[:77] + "..."
                suffix = f" — {result_text}"
            lines.append(f"  {i}. {icon} {task.description}{suffix}")

        return "\n".join(lines)

    def format_compact(self) -> str:
        """紧凑格式：一行总结 + 简要列表"""
        if self.count() == 0:
            return "（暂无子任务）"

        done = self.count_by_status(TaskStatus.DONE)
        in_progress = self.count_by_status(TaskStatus.IN_PROGRESS)
        pending = self.count_by_status(TaskStatus.PENDING)

        parts = [f"任务: {done}✅/{self.count()}总"]
        if in_progress > 0:
            parts.append(f"{in_progress}🔄进行中")
        if pending > 0:
            parts.append(f"{pending}⬜待处理")

        current = None
        for t in self._tasks:
            if t.status == TaskStatus.IN_PROGRESS:
                current = t.description
                break

        result = ", ".join(parts)
        if current:
            result += f" | 当前: {current}"
        return result

    def to_dict(self) -> dict:
        """序列化为字典（用于持久化）"""
        return {
            "tasks": [
                {
                    "description": t.description,
                    "status": t.status.value,
                    "result": t.result,
                }
                for t in self._tasks
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TodoList":
        """从字典反序列化"""
        todo = cls()
        for item in data.get("tasks", []):
            task = Task(
                description=item["description"],
                status=TaskStatus(item.get("status", "pending")),
                result=item.get("result"),
            )
            todo._tasks.append(task)
        return todo

    def __len__(self) -> int:
        return len(self._tasks)

    def __repr__(self) -> str:
        done = self.count_by_status(TaskStatus.DONE)
        return f"<TodoList tasks={self.count()} done={done}>"