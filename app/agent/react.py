from abc import ABC, abstractmethod

from pydantic import Field

from app.agent.base import BaseAgent
from app.llm import LLM
from app.schema import AgentState, Memory


class ReActAgent(BaseAgent, ABC):
    name: str
    description: str | None = None

    system_prompt: str | None = None
    next_step_prompt: str | None = None

    llm: LLM | None = Field(default_factory=LLM)
    memory: Memory = Field(default_factory=Memory)
    state: AgentState = AgentState.IDLE

    max_steps: int = 10
    current_step: int = 0

    @abstractmethod
    async def think(self) -> bool:
        """現在の状態を処理し、次のアクションを決定します"""

    @abstractmethod
    async def act(self) -> str:
        """決定されたアクションを実行します"""

    async def step(self) -> str:
        """単一のステップを実行します：思考と行動を行います。"""
        should_act = await self.think()
        if not should_act:
            return "思考完了 - アクション不要"
        return await self.act()
