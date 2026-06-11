import logging
from typing import Dict, List, Optional, Any
from tools.base_tool import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if not isinstance(tool, BaseTool):
            raise TypeError(
                f"register() 需要 BaseTool 实例，收到了 {type(tool).__name__}"
            )

        name = tool.name
        if not name:
            raise ValueError("Tool 必须定义 name 属性")

        if name in self._tools:
            old = self._tools[name]
            logger.warning(
                f"Tool '{name}' 已存在（{type(old).__name__}），"
                f"将被 {type(tool).__name__} 覆盖"
            )
        self._tools[name] = tool
        logger.info(f"Tool 已注册: {tool}")

    def unregister(self, name: str) -> Optional[BaseTool]:
        """
        注销指定名称的 Tool

        Args:
            name: Tool 名称

        Returns:
            被移除的 Tool 实例，如果不存在则返回 None
        """
        tool = self._tools.pop(name, None)
        if tool:
            logger.info(f"Tool 已注销: {tool}")
        else:
            logger.warning(f"尝试注销不存在的 Tool: '{name}'")
        return tool

    def get(self, name: str) -> Optional[BaseTool]:
        """
        按名称获取 Tool 实例

        Args:
            name: Tool 名称

        Returns:
            Tool 实例，不存在返回 None
        """
        return self._tools.get(name)

    # ── 查询 ────────────────────────────────────

    def list_names(self) -> List[str]:
        """
        获取所有已注册 Tool 的名称列表

        Returns:
            名称列表
        """
        return list(self._tools.keys())

    def list_all(self) -> List[BaseTool]:
        """
        获取所有已注册的 Tool 实例列表

        Returns:
            Tool 实例列表
        """
        return list(self._tools.values())

    # ── Function Calling 相关 ──────────────────

    def get_function_declarations(self) -> List[Dict[str, Any]]:
        return [tool.to_function_declaration() for tool in self._tools.values()]

    def execute(self, name: str, **kwargs) -> str:
        """
        按名称执行指定 Tool

        Args:
            name: Tool 名称
            **kwargs: 传递给 Tool.execute() 的参数

        Returns:
            Tool 执行结果字符串

        Raises:
            KeyError: Tool 名称不存在
        """
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(
                f"Tool '{name}' 未注册。可用工具: {self.list_names()}"
            )
        # 校验并执行
        validated = tool.validate_params(**kwargs)
        logger.debug(f"执行 Tool: {tool.name} with {list(validated.keys())}")
        return tool.execute(**validated)

    def clear(self) -> None:
        """清空所有已注册的 Tool"""
        count = len(self._tools)
        self._tools.clear()
        logger.info(f"已清空 {count} 个 Tool")

    @property
    def tool_count(self) -> int:
        """已注册的 Tool 数量"""
        return len(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        names = self.list_names()
        return f"<ToolRegistry tools={names}>"