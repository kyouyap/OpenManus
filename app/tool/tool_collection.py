"""複数のツールを管理するためのコレクションクラス。"""

from typing import Any

from app.exceptions import ToolError
from app.tool.base import BaseTool, ToolFailure, ToolResult


class ToolCollection:
    """定義されたツールのコレクション。"""

    def __init__(self, *tools: BaseTool):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

    def __iter__(self):
        return iter(self.tools)

    def to_params(self) -> list[dict[str, Any]]:
        return [tool.to_param() for tool in self.tools]

    async def execute(
        self, *, name: str, tool_input: dict[str, Any] = None
    ) -> ToolResult:
        tool = self.tool_map.get(name)
        if not tool:
            return ToolFailure(error=f"ツール {name} は無効です")
        try:
            result = await tool(**tool_input)
            return result
        except ToolError as e:
            return ToolFailure(error=e.message)

    async def execute_all(self) -> list[ToolResult]:
        """コレクション内の全てのツールを順番に実行します。"""
        results = []
        for tool in self.tools:
            try:
                result = await tool()
                results.append(result)
            except ToolError as e:
                results.append(ToolFailure(error=e.message))
        return results

    def get_tool(self, name: str) -> BaseTool:
        return self.tool_map.get(name)

    def add_tool(self, tool: BaseTool):
        self.tools += (tool,)
        self.tool_map[tool.name] = tool
        return self

    def add_tools(self, *tools: BaseTool):
        for tool in tools:
            self.add_tool(tool)
        return self
