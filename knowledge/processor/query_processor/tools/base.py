import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentTool:
    """Agent 可调用的工具基类。

    每个工具需要提供 name、description、parameters 以及实现 run 方法。
    tools_json_schema() 用于生成 function calling 的 schema。
    """

    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}

    def run(self, **kwargs) -> Any:
        raise NotImplementedError

    def tools_json_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def safe_run(self, **kwargs) -> str:
        try:
            result = self.run(**kwargs)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Tool {self.name} execution failed: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)


class ToolRegistry:
    """工具注册表：统一管理所有 Agent 可调用工具。"""

    _tools: Dict[str, AgentTool] = {}

    @classmethod
    def register(cls, tool: AgentTool):
        cls._tools[tool.name] = tool

    @classmethod
    def get(cls, name: str) -> Optional[AgentTool]:
        return cls._tools.get(name)

    @classmethod
    def all_schemas(cls) -> List[Dict[str, Any]]:
        return [t.tools_json_schema() for t in cls._tools.values()]

    @classmethod
    def clear(cls):
        cls._tools.clear()