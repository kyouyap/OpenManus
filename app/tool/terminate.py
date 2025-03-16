from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Any

from app.tool.base import BaseTool

_TERMINATE_DESCRIPTION = """リクエストが満たされた場合、またはアシスタントがタスクをこれ以上進められない場合に対話を終了します。
全てのタスクが完了したら、このツールを呼び出して作業を終了します。"""


class Terminate(BaseTool):
    name: str = "terminate"
    description: str = _TERMINATE_DESCRIPTION
    parameters: dict | None = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "対話の終了ステータス。",
                "enum": ["success", "failure"],
            }
        },
        "required": ["status"],
    }

    async def execute(self, **kwargs: Any) -> Any:
        """現在の実行を終了します"""
        status = kwargs.get("status")
        if not status:
            raise ValueError("statusパラメータが必要です")
        return f"対話を終了しました。ステータス: {status}"
