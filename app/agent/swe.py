from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.swe import NEXT_STEP_TEMPLATE, SYSTEM_PROMPT
from app.tool import Bash, StrReplaceEditor, Terminate, ToolCollection


class SWEAgent(ToolCallAgent):
    """コード実行と自然な対話のためのSWEAgentパラダイムを実装するエージェント。"""

    name: str = "swe"
    description: str = (
        "タスクを解決するためにコンピュータと直接対話する自律型AIプログラマー。"
    )

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_TEMPLATE

    available_tools: ToolCollection = ToolCollection(
        Bash(), StrReplaceEditor(), Terminate()
    )
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    max_steps: int = 30

    bash: Bash = Field(default_factory=Bash)
    working_dir: str = "."

    async def think(self) -> bool:
        """現在の状態を処理し、次のアクションを決定します"""
        # 作業ディレクトリを更新
        self.working_dir = await self.bash.execute("pwd")
        self.next_step_prompt = self.next_step_prompt.format(
            current_dir=self.working_dir
        )

        return await super().think()
