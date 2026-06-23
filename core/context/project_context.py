"""
每轮对话自动注入到 system prompt 中，让 Agent 始终遵循项目规则。
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ProjectContext:
    def __init__(self, path: str = "AGENT.md"):
        """
        Args:
            path: AGENT.md 文件路径（建议传入绝对路径以确保跨目录工作正常）
        """
        self._path = path
        self._content: Optional[str] = None
        self._loaded = False
        # 初始化时立即加载文件
        self.reload()

    @property
    def path(self) -> str:
        """AGENT.md 文件完整路径"""
        return os.path.abspath(self._path)

    @property
    def content(self) -> Optional[str]:
        """AGENT.md 原始内容"""
        return self._content

    @property
    def is_loaded(self) -> bool:
        """是否已成功加载内容"""
        return self._loaded and self._content is not None

    @property
    def size(self) -> int:
        """已加载内容的字符数（未加载时为 0）"""
        if self._content is None:
            return 0
        return len(self._content)

    def reload(self) -> bool:
        self._loaded = True
        if not os.path.exists(self._path):
            logger.debug(f"AGENT.md 不存在: {self._path}")
            self._content = None
            return False

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                self._content = f.read()
            if self._content.strip():
                logger.info(f"AGENT.md 已加载: {len(self._content)} 字符")
                return True
            else:
                logger.warning("AGENT.md 文件为空")
                self._content = None
                return False
        except Exception as e:
            logger.warning(f"AGENT.md 读取失败: {e}")
            self._content = None
            return False

    def get_system_prompt_section(self) -> str:
        if not self._content:
            return ""

        return (
            "【项目规则与约束 — 来自 AGENT.md】\n"
            "请严格遵循以下项目规则进行代码编写与操作：\n\n"
            f"{self._content.strip()}"
        )

    def get_context_str(self) -> Optional[str]:
        section = self.get_system_prompt_section()
        if not section:
            return None
        return section

    def __bool__(self) -> bool:
        """布尔判断：是否已成功加载内容"""
        return self.is_loaded

    def __repr__(self) -> str:
        status = "已加载" if self.is_loaded else "未加载"
        return f"<ProjectContext path='{self._path}' status={status} size={self.size}>"