from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel

from app.agent.base import BaseAgent


class FlowType(str, Enum):
    PLANNING = "planning"


class BaseFlow(BaseModel, ABC):
    """複数のエージェントをサポートする実行フローの基本クラス"""

    agents: dict[str, BaseAgent]
    tools: list | None = None
    primary_agent_key: str | None = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self, agents: BaseAgent | list[BaseAgent] | dict[str, BaseAgent], **data
    ):
        # エージェントの提供方法に応じた処理
        if isinstance(agents, BaseAgent):
            agents_dict = {"default": agents}
        elif isinstance(agents, list):
            agents_dict = {f"agent_{i}": agent for i, agent in enumerate(agents)}
        else:
            agents_dict = agents

        # プライマリエージェントが指定されていない場合、最初のエージェントを使用
        primary_key = data.get("primary_agent_key")
        if not primary_key and agents_dict:
            primary_key = next(iter(agents_dict))
            data["primary_agent_key"] = primary_key

        # エージェント辞書を設定
        data["agents"] = agents_dict

        # BaseModelの初期化を使用
        super().__init__(**data)

    @property
    def primary_agent(self) -> BaseAgent | None:
        """フローのプライマリエージェントを取得します"""
        return self.agents.get(self.primary_agent_key)

    def get_agent(self, key: str) -> BaseAgent | None:
        """キーを指定して特定のエージェントを取得します"""
        return self.agents.get(key)

    def add_agent(self, key: str, agent: BaseAgent) -> None:
        """フローに新しいエージェントを追加します"""
        self.agents[key] = agent

    @abstractmethod
    async def execute(self, input_text: str) -> str:
        """与えられた入力でフローを実行します"""


class PlanStepStatus(str, Enum):
    """計画ステップの可能な状態を定義する列挙クラス"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"

    @classmethod
    def get_all_statuses(cls) -> list[str]:
        """全ての可能なステップ状態値のリストを返します"""
        return [status.value for status in cls]

    @classmethod
    def get_active_statuses(cls) -> list[str]:
        """アクティブな状態（未開始または進行中）を表す値のリストを返します"""
        return [cls.NOT_STARTED.value, cls.IN_PROGRESS.value]

    @classmethod
    def get_status_marks(cls) -> dict[str, str]:
        """状態とそのマーカーシンボルのマッピングを返します"""
        return {
            cls.COMPLETED.value: "[✓]",
            cls.IN_PROGRESS.value: "[→]",
            cls.BLOCKED.value: "[!]",
            cls.NOT_STARTED.value: "[ ]",
        }
