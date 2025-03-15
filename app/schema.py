from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Role(str, Enum):
    """メッセージのロールオプション"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


ROLE_VALUES = tuple(role.value for role in Role)
ROLE_TYPE = Literal[ROLE_VALUES]  # type: ignore


class ToolChoice(str, Enum):
    """ツール選択オプション"""

    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


TOOL_CHOICE_VALUES = tuple(choice.value for choice in ToolChoice)
TOOL_CHOICE_TYPE = Literal[TOOL_CHOICE_VALUES]  # type: ignore


class AgentState(str, Enum):
    """エージェントの実行状態"""

    IDLE = "IDLE"  # アイドル状態
    RUNNING = "RUNNING"  # 実行中
    FINISHED = "FINISHED"  # 完了
    ERROR = "ERROR"  # エラー


class Function(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    """メッセージ内のツール/関数呼び出しを表現します"""

    id: str
    type: str = "function"
    function: Function


class Message(BaseModel):
    """会話内のチャットメッセージを表現します"""

    role: ROLE_TYPE = Field(...)  # type: ignore
    content: str | None = Field(default=None)
    tool_calls: list[ToolCall] | None = Field(default=None)
    name: str | None = Field(default=None)
    tool_call_id: str | None = Field(default=None)

    def __add__(self, other) -> list["Message"]:
        """Message + list または Message + Message の操作をサポートします"""
        if isinstance(other, list):
            return [self] + other
        if isinstance(other, Message):
            return [self, other]
        raise TypeError(
            f"サポートされていない演算子の型: '{type(self).__name__}' と '{type(other).__name__}'"
        )

    def __radd__(self, other) -> list["Message"]:
        """List + Message の操作をサポートします"""
        if isinstance(other, list):
            return other + [self]
        raise TypeError(
            f"サポートされていない演算子の型: '{type(other).__name__}' と '{type(self).__name__}'"
        )

    def to_dict(self) -> dict:
        """メッセージを辞書形式に変換します"""
        message = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls is not None:
            message["tool_calls"] = [tool_call.dict() for tool_call in self.tool_calls]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        return message

    @classmethod
    def user_message(cls, content: str) -> "Message":
        """ユーザーメッセージを作成します"""
        return cls(role=Role.USER, content=content)

    @classmethod
    def system_message(cls, content: str) -> "Message":
        """システムメッセージを作成します"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_message(cls, content: str | None = None) -> "Message":
        """アシスタントメッセージを作成します"""
        return cls(role=Role.ASSISTANT, content=content)

    @classmethod
    def tool_message(cls, content: str, name, tool_call_id: str) -> "Message":
        """ツールメッセージを作成します"""
        return cls(
            role=Role.TOOL, content=content, name=name, tool_call_id=tool_call_id
        )

    @classmethod
    def from_tool_calls(
        cls, tool_calls: list[Any], content: str | list[str] = "", **kwargs
    ):
        """生のツール呼び出しからToolCallsMessageを作成します。

        引数:
            tool_calls: LLMからの生のツール呼び出し
            content: オプションのメッセージ内容
        """
        formatted_calls = [
            {"id": call.id, "function": call.function.model_dump(), "type": "function"}
            for call in tool_calls
        ]
        return cls(
            role=Role.ASSISTANT, content=content, tool_calls=formatted_calls, **kwargs
        )


class Memory(BaseModel):
    messages: list[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)

    def add_message(self, message: Message) -> None:
        """メッセージをメモリに追加します"""
        self.messages.append(message)
        # オプション: メッセージ制限を実装
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_messages(self, messages: list[Message]) -> None:
        """複数のメッセージをメモリに追加します"""
        self.messages.extend(messages)

    def clear(self) -> None:
        """全てのメッセージをクリアします"""
        self.messages.clear()

    def get_recent_messages(self, n: int) -> list[Message]:
        """最新のn件のメッセージを取得します"""
        return self.messages[-n:]

    def to_dict_list(self) -> list[dict]:
        """メッセージを辞書のリストに変換します"""
        return [msg.to_dict() for msg in self.messages]
