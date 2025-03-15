from app.tool.base import BaseTool

_TERMINATE_DESCRIPTION = """リクエストが満たされた場合、またはアシスタントがタスクをこれ以上進められない場合に対話を終了します。
全てのタスクが完了したら、このツールを呼び出して作業を終了します。"""


class Terminate(BaseTool):
    name: str = "terminate"
    description: str = _TERMINATE_DESCRIPTION
    parameters: dict = {
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

    async def execute(self, status: str) -> str:
        """現在の実行を終了します"""
        return f"対話を終了しました。ステータス: {status}"
