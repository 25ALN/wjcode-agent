from typing import Dict, Any
from abc import ABC, abstractmethod


SAFE = "safe"           # 自动执行，无需确认
CAUTION = "caution"     # 需用户确认
DANGEROUS = "dangerous" # 必须确认 + 显示警告


class BaseTool(ABC):

    name: str            # 工具名称（唯一标识，如 "read_file"）
    description: str     # 工具描述（告诉 LLM 何时该调用此工具）
    parameters: dict     # 参数 schema（JSON Schema 格式，描述 execute 的入参）
    risk_level: str = SAFE   # 权限等级：safe / caution / dangerous

    def execute(self, **kwargs) -> str:
        raise NotImplementedError(
            f"Tool '{self.name}' 未实现 execute() 方法"
        )

    def to_function_declaration(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def validate_params(self, **kwargs) -> Dict[str, Any]:
        """检查必填参数是否齐全，返回校验后的参数字典"""
        required = self.parameters.get("required", [])
        missing = [r for r in required if r not in kwargs or kwargs[r] is None]
        if missing:
            raise ValueError(
                f"Tool '{self.name}' 缺少必填参数: {', '.join(missing)}"
            )
        return kwargs

    def __repr__(self) -> str:
        required = self.parameters.get("required", [])
        return f"<{self.__class__.__name__} name='{self.name}' risk={self.risk_level} params={required}>"