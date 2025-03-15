from typing import Any

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.file_saver import FileSaver
from app.tool.python_execute import PythonExecute
from app.tool.web_search import WebSearch


class Manus(ToolCallAgent):
    """計画を使用して様々なタスクを解決する汎用エージェント。

    このエージェントはPlanningAgentを拡張し、Pythonの実行、Webブラウジング、
    ファイル操作、情報検索など、幅広いユーザーリクエストを処理するための
    包括的なツールと機能を備えています。
    """

    name: str = "Manus"
    description: str = "複数のツールを使用して様々なタスクを解決できる汎用エージェント"

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 2000
    max_steps: int = 20

    # 汎用ツールをツールコレクションに追加
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(), WebSearch(), BrowserUseTool(), FileSaver(), Terminate()
        )
    )

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        if not self._is_special_tool(name):
            return
        await self.available_tools.get_tool(BrowserUseTool().name).cleanup()
        await super()._handle_special_tool(name, result, **kwargs)
