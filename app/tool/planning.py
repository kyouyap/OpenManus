# tool/planning.py
from typing import Literal

from app.exceptions import ToolError
from app.tool.base import BaseTool, ToolResult

_PLANNING_TOOL_DESCRIPTION = """
複雑なタスクを解決するための計画を作成・管理できるプランニングツールです。
計画の作成、ステップの更新、進捗の追跡などの機能を提供します。
"""


class PlanningTool(BaseTool):
    """複雑なタスクを解決するための計画を作成・管理できるプランニングツールです。
    計画の作成、ステップの更新、進捗の追跡などの機能を提供します。
    """

    name: str = "planning"
    description: str = _PLANNING_TOOL_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "description": "実行するコマンド。利用可能なコマンド: create（作成）, update（更新）, list（一覧）, get（取得）, set_active（アクティブ設定）, mark_step（ステップ更新）, delete（削除）",
                "enum": [
                    "create",
                    "update",
                    "list",
                    "get",
                    "set_active",
                    "mark_step",
                    "delete",
                ],
                "type": "string",
            },
            "plan_id": {
                "description": "計画の一意の識別子。create, update, set_active, deleteコマンドで必須。getとmark_stepでは省略可能（省略時はアクティブな計画を使用）。",
                "type": "string",
            },
            "title": {
                "description": "計画のタイトル。createコマンドで必須、updateコマンドでは省略可能。",
                "type": "string",
            },
            "steps": {
                "description": "計画のステップリスト。createコマンドで必須、updateコマンドでは省略可能。",
                "type": "array",
                "items": {"type": "string"},
            },
            "step_index": {
                "description": "更新するステップのインデックス（0から開始）。mark_stepコマンドで必須。",
                "type": "integer",
            },
            "step_status": {
                "description": "ステップに設定するステータス。mark_stepコマンドで使用。",
                "enum": ["not_started", "in_progress", "completed", "blocked"],
                "type": "string",
            },
            "step_notes": {
                "description": "ステップの追加メモ。mark_stepコマンドでは省略可能。",
                "type": "string",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    plans: dict = {}  # Dictionary to store plans by plan_id
    _current_plan_id: str | None = None  # Track the current active plan

    async def execute(
        self,
        *,
        command: Literal[
            "create", "update", "list", "get", "set_active", "mark_step", "delete"
        ],
        plan_id: str | None = None,
        title: str | None = None,
        steps: list[str] | None = None,
        step_index: int | None = None,
        step_status: Literal["not_started", "in_progress", "completed", "blocked"]
        | None = None,
        step_notes: str | None = None,
        **kwargs,
    ):
        """指定されたコマンドとパラメータでプランニングツールを実行します。

        パラメータ:
        - command: 実行する操作
        - plan_id: 計画の一意の識別子
        - title: 計画のタイトル（createコマンドで使用）
        - steps: 計画のステップリスト（createコマンドで使用）
        - step_index: 更新するステップのインデックス（mark_stepコマンドで使用）
        - step_status: ステップに設定するステータス（mark_stepコマンドで使用）
        - step_notes: ステップの追加メモ（mark_stepコマンドで使用）
        """
        if command == "create":
            return self._create_plan(plan_id, title, steps)
        if command == "update":
            return self._update_plan(plan_id, title, steps)
        if command == "list":
            return self._list_plans()
        if command == "get":
            return self._get_plan(plan_id)
        if command == "set_active":
            return self._set_active_plan(plan_id)
        if command == "mark_step":
            return self._mark_step(plan_id, step_index, step_status, step_notes)
        if command == "delete":
            return self._delete_plan(plan_id)
        raise ToolError(
            f"認識できないコマンドです: {command}。使用可能なコマンド: create, update, list, get, set_active, mark_step, delete"
        )

    def _create_plan(
        self, plan_id: str | None, title: str | None, steps: list[str] | None
    ) -> ToolResult:
        """指定されたID、タイトル、ステップで新しい計画を作成します。"""
        if not plan_id:
            raise ToolError("createコマンドには`plan_id`パラメータが必要です")

        if plan_id in self.plans:
            raise ToolError(
                f"ID '{plan_id}' の計画は既に存在します。既存の計画を修正するにはupdateを使用してください。"
            )

        if not title:
            raise ToolError("createコマンドには`title`パラメータが必要です")

        if (
            not steps
            or not isinstance(steps, list)
            or not all(isinstance(step, str) for step in steps)
        ):
            raise ToolError(
                "createコマンドの`steps`パラメータは空でない文字列のリストである必要があります"
            )

        # Create a new plan with initialized step statuses
        plan = {
            "plan_id": plan_id,
            "title": title,
            "steps": steps,
            "step_statuses": ["not_started"] * len(steps),
            "step_notes": [""] * len(steps),
        }

        self.plans[plan_id] = plan
        self._current_plan_id = plan_id  # Set as active plan

        return ToolResult(
            output=f"ID: {plan_id} で計画を作成しました\n\n{self._format_plan(plan)}"
        )

    def _update_plan(
        self, plan_id: str | None, title: str | None, steps: list[str] | None
    ) -> ToolResult:
        """既存の計画を新しいタイトルまたはステップで更新します。"""
        if not plan_id:
            raise ToolError("updateコマンドには`plan_id`パラメータが必要です")

        if plan_id not in self.plans:
            raise ToolError(f"ID: {plan_id} の計画が見つかりません")

        plan = self.plans[plan_id]

        if title:
            plan["title"] = title

        if steps:
            if not isinstance(steps, list) or not all(
                isinstance(step, str) for step in steps
            ):
                raise ToolError(
                    "updateコマンドの`steps`パラメータは文字列のリストである必要があります"
                )

            # Preserve existing step statuses for unchanged steps
            old_steps = plan["steps"]
            old_statuses = plan["step_statuses"]
            old_notes = plan["step_notes"]

            # Create new step statuses and notes
            new_statuses = []
            new_notes = []

            for i, step in enumerate(steps):
                # If the step exists at the same position in old steps, preserve status and notes
                if i < len(old_steps) and step == old_steps[i]:
                    new_statuses.append(old_statuses[i])
                    new_notes.append(old_notes[i])
                else:
                    new_statuses.append("not_started")
                    new_notes.append("")

            plan["steps"] = steps
            plan["step_statuses"] = new_statuses
            plan["step_notes"] = new_notes

        return ToolResult(
            output=f"計画を更新しました: {plan_id}\n\n{self._format_plan(plan)}"
        )

    def _list_plans(self) -> ToolResult:
        """利用可能な全ての計画を一覧表示します。"""
        if not self.plans:
            return ToolResult(
                output="計画が存在しません。createコマンドで計画を作成してください。"
            )

        output = "利用可能な計画:\n"
        for plan_id, plan in self.plans.items():
            current_marker = " (アクティブ)" if plan_id == self._current_plan_id else ""
            completed = sum(
                1 for status in plan["step_statuses"] if status == "completed"
            )
            total = len(plan["steps"])
            progress = f"{completed}/{total} ステップ完了"
            output += f"• {plan_id}{current_marker}: {plan['title']} - {progress}\n"

        return ToolResult(output=output)

    def _get_plan(self, plan_id: str | None) -> ToolResult:
        """特定の計画の詳細を取得します。"""
        if not plan_id:
            # If no plan_id is provided, use the current active plan
            if not self._current_plan_id:
                raise ToolError(
                    "アクティブな計画がありません。plan_idを指定するかアクティブな計画を設定してください。"
                )
            plan_id = self._current_plan_id

        if plan_id not in self.plans:
            raise ToolError(f"ID: {plan_id} の計画が見つかりません")

        plan = self.plans[plan_id]
        return ToolResult(output=self._format_plan(plan))

    def _set_active_plan(self, plan_id: str | None) -> ToolResult:
        """計画をアクティブな計画として設定します。"""
        if not plan_id:
            raise ToolError("set_activeコマンドには`plan_id`パラメータが必要です")

        if plan_id not in self.plans:
            raise ToolError(f"No plan found with ID: {plan_id}")

        self._current_plan_id = plan_id
        return ToolResult(
            output=f"'{plan_id}' をアクティブな計画に設定しました。\n\n{self._format_plan(self.plans[plan_id])}"
        )

    def _mark_step(
        self,
        plan_id: str | None,
        step_index: int | None,
        step_status: str | None,
        step_notes: str | None,
    ) -> ToolResult:
        """ステップを特定のステータスと任意のメモで更新します。"""
        if not plan_id:
            # If no plan_id is provided, use the current active plan
            if not self._current_plan_id:
                raise ToolError(
                    "アクティブな計画がありません。plan_idを指定するかアクティブな計画を設定してください。"
                )
            plan_id = self._current_plan_id

        if plan_id not in self.plans:
            raise ToolError(f"No plan found with ID: {plan_id}")

        if step_index is None:
            raise ToolError("mark_stepコマンドには`step_index`パラメータが必要です")

        plan = self.plans[plan_id]

        if step_index < 0 or step_index >= len(plan["steps"]):
            raise ToolError(
                f"無効なstep_index: {step_index}。有効な範囲は0から{len(plan['steps']) - 1}です。"
            )

        if step_status and step_status not in [
            "not_started",
            "in_progress",
            "completed",
            "blocked",
        ]:
            raise ToolError(
                f"無効なstep_status: {step_status}。有効なステータス: not_started, in_progress, completed, blocked"
            )

        if step_status:
            plan["step_statuses"][step_index] = step_status

        if step_notes:
            plan["step_notes"][step_index] = step_notes

        return ToolResult(
            output=f"計画 '{plan_id}' のステップ {step_index} を更新しました。\n\n{self._format_plan(plan)}"
        )

    def _delete_plan(self, plan_id: str | None) -> ToolResult:
        """計画を削除します。"""
        if not plan_id:
            raise ToolError("deleteコマンドには`plan_id`パラメータが必要です")

        if plan_id not in self.plans:
            raise ToolError(f"ID: {plan_id} の計画が見つかりません")

        del self.plans[plan_id]

        # If the deleted plan was the active plan, clear the active plan
        if self._current_plan_id == plan_id:
            self._current_plan_id = None

        return ToolResult(output=f"計画 '{plan_id}' を削除しました。")

    def _format_plan(self, plan: dict) -> str:
        """計画を表示用にフォーマットします。"""
        output = f"計画: {plan['title']} (ID: {plan['plan_id']})\n"
        output += "=" * len(output) + "\n\n"

        # Calculate progress statistics
        total_steps = len(plan["steps"])
        completed = sum(1 for status in plan["step_statuses"] if status == "completed")
        in_progress = sum(
            1 for status in plan["step_statuses"] if status == "in_progress"
        )
        blocked = sum(1 for status in plan["step_statuses"] if status == "blocked")
        not_started = sum(
            1 for status in plan["step_statuses"] if status == "not_started"
        )

        output += f"進捗状況: {completed}/{total_steps} ステップ完了 "
        if total_steps > 0:
            percentage = (completed / total_steps) * 100
            output += f"({percentage:.1f}%)\n"
        else:
            output += "(0%)\n"

        output += f"状態: {completed} 完了, {in_progress} 進行中, {blocked} ブロック中, {not_started} 未着手\n\n"
        output += "ステップ:\n"

        # Add each step with its status and notes
        for i, (step, status, notes) in enumerate(
            zip(plan["steps"], plan["step_statuses"], plan["step_notes"], strict=False)
        ):
            status_symbol = {
                "not_started": "[ ]",
                "in_progress": "[→]",
                "completed": "[✓]",
                "blocked": "[!]",
            }.get(status, "[ ]")

            output += f"{i}. {status_symbol} {step}\n"
            if notes:
                output += f"   メモ: {notes}\n"

        return output
