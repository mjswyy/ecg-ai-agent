"""
工具注册表 — 注册、发现和调用诊断工具 / Tool Registry for Agent tools.

中央注册表管理 Agent 可用的所有工具。每个工具有名称、描述、
参数 schema 和可调用的实现函数。

Central registry for all tools available to the AI Agent.
Each tool is defined by: name, function, description, parameter schema.

使用示例 / Usage:
    registry = ToolRegistry()
    registry.register("extract_r_peaks", r_peak_fn, description="检测R峰")
    result = registry.call("extract_r_peaks", ecg_signal=..., fs=500)
"""

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Agent 工具注册表 / Central tool registry.

    每个工具的定义包含:
        - name:         唯一标识符 / Unique identifier
        - func:         可调用实现 / Callable implementation
        - description:  人类可读描述 / Human-readable description
        - schema:       参数 schema (OpenAI function-calling 格式)
        - dependencies: 依赖的其他工具名列表 / List of tool names this depends on
    """

    def __init__(self):
        self._tools: Dict[str, Dict] = {}

    def register(self, name: str, func: Callable, description: str = "",
                 schema: Optional[Dict] = None, dependencies: Optional[List[str]] = None):
        """注册一个工具 / Register a tool.

        参数 / Args:
            name:         工具名 / Tool name.
            func:         可调用函数 / Callable.
            description:  功能描述 / What the tool does.
            schema:       参数定义 / OpenAI-style parameter schema.
            dependencies: 依赖列表 / List of prerequisite tool names.
        """
        self._tools[name] = {
            "name": name, "func": func, "description": description,
            "schema": schema or {}, "dependencies": dependencies or [],
        }
        logger.debug(f"工具已注册: {name}")

    def get(self, name: str) -> Optional[Dict]:
        """按名称获取工具 / Get tool by name."""
        return self._tools.get(name)

    def call(self, name: str, **kwargs) -> Any:
        """按名称调用工具 / Invoke a tool by name.

        异常 / Raises:
            ValueError: 如果工具未找到 / If tool not found.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"工具未找到: {name}。可用: {self.list_names()}")
        try:
            return tool["func"](**kwargs)
        except Exception as e:
            logger.error(f"工具 '{name}' 执行失败: {e}")
            raise

    def list_tools(self) -> List[Dict]:
        """列出所有工具及其描述（供 LLM 选择）/ List all tools for LLM."""
        return [{"name": t["name"], "description": t["description"], "parameters": t["schema"]}
                for t in self._tools.values()]

    def list_names(self) -> List[str]:
        """列出所有注册的工具名 / List all registered tool names."""
        return list(self._tools.keys())

    def get_dependencies(self, name: str) -> List[str]:
        """获取工具的依赖列表 / Get dependency tool names."""
        tool = self._tools.get(name)
        return tool["dependencies"] if tool else []

    def __contains__(self, name: str) -> bool: return name in self._tools
    def __len__(self) -> int:                return len(self._tools)
