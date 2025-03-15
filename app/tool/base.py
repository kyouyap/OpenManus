from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, Field


class BaseTool(ABC, BaseModel):
    """ツールの基本クラス。"""

    name: str
    description: str
    parameters: ClassVar[dict | None] = None

    class Config:
        arbitrary_types_allowed = True

    async def __call__(self, **kwargs) -> Any:
        """与えられたパラメータでツールを実行します。"""
        return await self.execute(**kwargs)

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """与えられたパラメータでツールを実行します。"""

    def to_param(self) -> dict:
        """ツールを関数呼び出し形式に変換します。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolResult(BaseModel):
    """ツール実行の結果を表現します。"""

    output: Any = Field(default=None)
    error: str | None = Field(default=None)
    system: str | None = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __bool__(self):
        return any(getattr(self, field) for field in self.__fields__)

    def __add__(self, other: "ToolResult"):
        def combine_fields(
            field: str | None, other_field: str | None, concatenate: bool = True
        ):
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("ツール結果を結合できません")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            system=combine_fields(self.system, other.system),
        )

    def __str__(self):
        return f"エラー: {self.error}" if self.error else self.output

    def replace(self, **kwargs):
        """指定されたフィールドを置き換えた新しいToolResultを返します。"""
        return type(self)(**{**self.dict(), **kwargs})


class CLIResult(ToolResult):
    """CLIの出力として表示可能なToolResult。"""


class ToolFailure(ToolResult):
    """失敗を表すToolResult。"""


class AgentAwareTool:
    """エージェントを認識するツール。"""

    agent: Optional = None
