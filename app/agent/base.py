from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any, Protocol, TypeVar, overload

from pydantic import BaseModel, Field, model_validator

from app.llm import LLM
from app.logger import logger
from app.schema import ROLE_TYPE, AgentState, Memory, Message, Role

T = TypeVar("T")


class BasicMessageFactory(Protocol):
    """基本的なメッセージファクトリー関数の型プロトコル"""

    def __call__(self, content: str) -> Message: ...


@overload
def ensure_message_factory(func: Callable[[str], Message]) -> BasicMessageFactory: ...


@overload
def ensure_message_factory(
    func: Callable[[str, str, str], Message],
) -> Callable[[str, str, str], Message]: ...


def ensure_message_factory(func: Any) -> Any:
    """関数をMessageFactoryとして扱えるようにラップします"""
    return func


class BaseAgent(BaseModel, ABC):
    """エージェントの状態と実行を管理する抽象基底クラス。

    状態遷移、メモリ管理、ステップベースの実行ループのための基本機能を提供します。
    サブクラスは`step`メソッドを実装する必要があります。
    """

    # Core attributes
    name: str = Field(..., description="エージェントの一意な名前")
    description: str | None = Field(None, description="エージェントの説明（任意）")

    # プロンプト
    system_prompt: str | None = Field(
        None, description="システムレベルの指示プロンプト"
    )
    next_step_prompt: str | None = Field(
        None, description="次のアクションを決定するためのプロンプト"
    )

    # 依存関係
    llm: LLM | None = Field(None, description="言語モデルのインスタンス")
    memory: Memory = Field(
        default_factory=Memory, description="エージェントのメモリストア"
    )
    state: AgentState = Field(
        default=AgentState.IDLE, description="現在のエージェント状態"
    )

    # 実行制御
    max_steps: int = Field(default=10, description="終了までの最大ステップ数")
    current_step: int = Field(default=0, description="実行中の現在のステップ")

    duplicate_threshold: int = 2

    class Config:
        """Pydanticモデルの設定オプション。"""

        arbitrary_types_allowed = True
        extra = "allow"  # サブクラスの柔軟性のために追加フィールドを許可

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """提供されていない場合、デフォルト設定でエージェントを初期化します。"""
        if self.llm is None:
            self.llm = LLM(config_name=self.name.lower())
        if not isinstance(self.memory, Memory):
            self.memory = Memory()
        return self

    @asynccontextmanager
    async def state_context(self, new_state: AgentState) -> AsyncGenerator[None, None]:
        """安全なエージェント状態遷移のためのコンテキストマネージャー。

        Args:
            new_state: コンテキスト中に遷移する状態。

        Yields:
            None: 新しい状態での実行を可能にします。

        Raises:
            ValueError: new_stateが無効な場合。

        """
        if not isinstance(new_state, AgentState):
            error_message = f"Invalid state: {new_state}"
            raise TypeError(error_message)

        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR  # Transition to ERROR on failure
            raise e
        finally:
            self.state = previous_state  # Revert to previous state

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        **kwargs: Any,
    ) -> None:
        """エージェントのメモリにメッセージを追加します。

        Args:
            role: メッセージ送信者の役割（user、system、assistant、tool）。
            content: メッセージの内容。
            **kwargs: 追加の引数（例：toolメッセージのtool_call_id）。

        Raises:
            ValueError: roleが未対応の場合。

        """
        message_factories = {
            Role.USER.value: ensure_message_factory(Message.user_message),
            Role.SYSTEM.value: ensure_message_factory(Message.system_message),
            Role.ASSISTANT.value: ensure_message_factory(Message.assistant_message),
        }

        if role not in message_factories and role != Role.TOOL.value:
            raise ValueError(f"Unsupported message role: {role}")

        if role == Role.TOOL.value:
            if "name" not in kwargs or "tool_call_id" not in kwargs:
                raise ValueError("Tool messages require 'name' and 'tool_call_id'")
            msg = Message.tool_message(
                content=content,
                name=kwargs["name"],
                tool_call_id=kwargs["tool_call_id"],
            )
        else:
            msg_factory = message_factories[role]
            msg = msg_factory(content)

        self.memory.add_message(msg)

    async def run(self, request: str | None = None) -> str:
        """エージェントのメインループを非同期で実行します。

        Args:
            request: 処理する初期ユーザーリクエスト（任意）。

        Returns:
            実行結果を要約した文字列。

        Raises:
            RuntimeError: 開始時にエージェントがIDLE状態でない場合。

        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.update_memory(Role.USER.value, request)

        results: list[str] = []
        async with self.state_context(AgentState.RUNNING):
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                logger.info(f"Executing step {self.current_step}/{self.max_steps}")
                step_result = await self.step()

                # Check for stuck state
                if self.is_stuck():
                    self._handle_stuck_state()

                results.append(f"Step {self.current_step}: {step_result}")

            if self.current_step >= self.max_steps:
                self.current_step = 0
                self.state = AgentState.IDLE
                results.append(f"Terminated: Reached max steps ({self.max_steps})")

        return "\n".join(results) if results else "No steps executed"

    @abstractmethod
    async def step(self) -> str:
        """エージェントのワークフローで単一ステップを実行します。

        サブクラスで具体的な動作を定義するために実装する必要があります。

        Returns:
            str: ステップの実行結果を示す文字列

        """
        raise NotImplementedError

    def _handle_stuck_state(self) -> None:
        """戦略を変更するプロンプトを追加してスタック状態を処理します"""
        stuck_prompt = (
            "重複した応答が検出されました。新しい戦略を検討し、"
            "すでに試みて効果のなかった方法の繰り返しを避けてください。"
        )
        if self.next_step_prompt:
            self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(
            f"エージェントがスタック状態を検出しました。プロンプトを追加: {stuck_prompt}"
        )

    def is_stuck(self) -> bool:
        """重複コンテンツを検出してエージェントがループでスタックしているかを確認します"""
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        # Count identical content occurrences
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == Role.ASSISTANT.value and msg.content == last_message.content
        )

        return duplicate_count >= self.duplicate_threshold

    @property
    def messages(self) -> list[Message]:
        """エージェントのメモリからメッセージのリストを取得します。"""
        return self.memory.messages

    @messages.setter
    def messages(self, value: list[Message]) -> None:
        """エージェントのメモリにメッセージのリストを設定します。"""
        self.memory.messages = value
