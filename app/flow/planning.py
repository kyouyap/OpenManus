import json
import time

from pydantic import Field

from app.agent.base import BaseAgent
from app.flow.base import BaseFlow, PlanStepStatus
from app.llm import LLM
from app.logger import logger
from app.schema import AgentState, Message, ToolChoice
from app.tool import PlanningTool


class PlanningFlow(BaseFlow):
    """エージェントを使用してタスクの計画と実行を管理するフロー。"""

    llm: LLM = Field(default_factory=lambda: LLM())
    planning_tool: PlanningTool = Field(default_factory=PlanningTool)
    executor_keys: list[str] = Field(default_factory=list)
    active_plan_id: str = Field(default_factory=lambda: f"plan_{int(time.time())}")
    current_step_index: int | None = None

    def __init__(
        self, agents: BaseAgent | list[BaseAgent] | dict[str, BaseAgent], **data
    ):
        # super().__init__の前にexecutor_keysを設定
        if "executors" in data:
            data["executor_keys"] = data.pop("executors")

        # プランIDが提供された場合は設定
        if "plan_id" in data:
            data["active_plan_id"] = data.pop("plan_id")

        # 提供されていない場合はプランニングツールを初期化
        if "planning_tool" not in data:
            planning_tool = PlanningTool()
            data["planning_tool"] = planning_tool

        # 処理済みのデータで親の初期化を呼び出し
        super().__init__(agents, **data)

        # executor_keysが指定されていない場合、すべてのエージェントキーを設定
        if not self.executor_keys:
            self.executor_keys = list(self.agents.keys())

    def get_executor(self, step_type: str | None = None) -> BaseAgent:
        """現在のステップに適切な実行エージェントを取得します。
        ステップのタイプ/要件に基づいてエージェントを選択するように拡張可能です。
        """
        # ステップタイプが提供され、エージェントキーと一致する場合、そのエージェントを使用
        if step_type and step_type in self.agents:
            return self.agents[step_type]

        # それ以外の場合、最初の利用可能な実行者を使用するかプライマリエージェントにフォールバック
        for key in self.executor_keys:
            if key in self.agents:
                return self.agents[key]

        # プライマリエージェントにフォールバック
        return self.primary_agent

    async def execute(self, input_text: str) -> str:
        """エージェントを使用してプランニングフローを実行します。"""
        try:
            if not self.primary_agent:
                raise ValueError("利用可能なプライマリエージェントがありません")

            # 入力が提供された場合、初期計画を作成
            if input_text:
                await self._create_initial_plan(input_text)

                # 計画が正常に作成されたことを確認
                if self.active_plan_id not in self.planning_tool.plans:
                    logger.error(
                        f"計画の作成に失敗しました。プランID {self.active_plan_id} がプランニングツールに見つかりません。"
                    )
                    return f"以下の計画の作成に失敗しました: {input_text}"

            result = ""
            while True:
                # 実行する現在のステップを取得
                self.current_step_index, step_info = await self._get_current_step_info()

                # ステップがない場合または計画が完了した場合は終了
                if self.current_step_index is None:
                    result += await self._finalize_plan()
                    break

                # 適切なエージェントで現在のステップを実行
                step_type = step_info.get("type") if step_info else None
                executor = self.get_executor(step_type)
                step_result = await self._execute_step(executor, step_info)
                result += step_result + "\n"

                # エージェントが終了を望んでいるかチェック
                if hasattr(executor, "state") and executor.state == AgentState.FINISHED:
                    break

            return result
        except Exception as e:
            logger.error(f"PlanningFlowでエラーが発生しました: {e!s}")
            return f"実行に失敗しました: {e!s}"

    async def _create_initial_plan(self, request: str) -> None:
        """フローのLLMとPlanningToolを使用して、リクエストに基づいて初期計画を作成します。"""
        logger.info(f"ID: {self.active_plan_id} で初期計画を作成中")

        # 計画作成用のシステムメッセージを作成
        system_message = Message.system_message(
            "あなたは計画立案アシスタントです。簡潔で実行可能な計画を、明確なステップで作成してください。"
            "詳細なサブステップではなく、主要なマイルストーンに焦点を当ててください。"
            "明確さと効率性を最適化してください。"
        )

        # リクエストを含むユーザーメッセージを作成
        user_message = Message.user_message(
            f"タスクを達成するための合理的な計画を、明確なステップで作成してください: {request}"
        )

        # PlanningToolを使用してLLMを呼び出し
        response = await self.llm.ask_tool(
            messages=[user_message],
            system_msgs=[system_message],
            tools=[self.planning_tool.to_param()],
            tool_choice=ToolChoice.AUTO,
        )

        # ツール呼び出しが存在する場合は処理
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name == "planning":
                    # 引数をパース
                    args = tool_call.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            logger.error(f"ツール引数のパースに失敗しました: {args}")
                            continue

                    # プランIDが正しく設定されていることを確認し、ツールを実行
                    args["plan_id"] = self.active_plan_id

                    # ツールを直接ではなくToolCollectionを介して実行
                    result = await self.planning_tool.execute(**args)

                    logger.info(f"計画作成結果: {result!s}")
                    return

        # ここまで実行が到達した場合はデフォルトの計画を作成
        logger.warning("デフォルトの計画を作成中")

        # ToolCollectionを使用してデフォルトの計画を作成
        await self.planning_tool.execute(
            command="create",
            plan_id=self.active_plan_id,
            title=f"計画: {request[:50]}{'...' if len(request) > 50 else ''}",
            steps=["リクエストの分析", "タスクの実行", "結果の検証"],
        )

    async def _get_current_step_info(self) -> tuple[int | None, dict | None]:
        """現在の計画を解析して、最初の未完了ステップのインデックスと情報を特定します。
        アクティブなステップが見つからない場合は(None, None)を返します。
        """
        if (
            not self.active_plan_id
            or self.active_plan_id not in self.planning_tool.plans
        ):
            logger.error(f"プランID {self.active_plan_id} が見つかりません")
            return None, None

        try:
            # プランニングツールのストレージから計画データに直接アクセス
            plan_data = self.planning_tool.plans[self.active_plan_id]
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])

            # 最初の未完了ステップを探す
            for i, step in enumerate(steps):
                if i >= len(step_statuses):
                    status = PlanStepStatus.NOT_STARTED.value
                else:
                    status = step_statuses[i]

                if status in PlanStepStatus.get_active_statuses():
                    # ステップタイプ/カテゴリが利用可能な場合は抽出
                    step_info = {"text": step}

                    # テキストからステップタイプを抽出（例：[SEARCH]や[CODE]）
                    import re

                    type_match = re.search(r"\[([A-Z_]+)\]", step)
                    if type_match:
                        step_info["type"] = type_match.group(1).lower()

                    # 現在のステップをin_progressとしてマーク
                    try:
                        await self.planning_tool.execute(
                            command="mark_step",
                            plan_id=self.active_plan_id,
                            step_index=i,
                            step_status=PlanStepStatus.IN_PROGRESS.value,
                        )
                    except Exception as e:
                        logger.warning(
                            f"ステップをin_progressとしてマークする際にエラーが発生しました: {e}"
                        )
                        # 必要に応じてステップステータスを直接更新
                        if i < len(step_statuses):
                            step_statuses[i] = PlanStepStatus.IN_PROGRESS.value
                        else:
                            while len(step_statuses) < i:
                                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                            step_statuses.append(PlanStepStatus.IN_PROGRESS.value)

                        plan_data["step_statuses"] = step_statuses

                    return i, step_info

            return None, None  # アクティブなステップが見つかりません

        except Exception as e:
            logger.warning(
                f"現在のステップインデックスの検索中にエラーが発生しました: {e}"
            )
            return None, None

    async def _execute_step(self, executor: BaseAgent, step_info: dict) -> str:
        """agent.run()を使用して、指定されたエージェントで現在のステップを実行します。"""
        # 現在の計画状態でエージェントのコンテキストを準備
        plan_status = await self._get_plan_text()
        step_text = step_info.get("text", f"ステップ {self.current_step_index}")

        # 現在のステップを実行するためのエージェントのプロンプトを作成
        step_prompt = f"""
        現在の計画状態:
        {plan_status}

        あなたの現在のタスク:
        あなたは現在、ステップ {self.current_step_index} に取り組んでいます: "{step_text}"

        適切なツールを使用してこのステップを実行してください。完了したら、達成したことの要約を提供してください。
        """

        # agent.run()を使用してステップを実行
        try:
            step_result = await executor.run(step_prompt)

            # 実行が成功したらステップを完了としてマーク
            await self._mark_step_completed()

            return step_result
        except Exception as e:
            logger.error(
                f"ステップ {self.current_step_index} の実行中にエラーが発生しました: {e}"
            )
            return f"ステップ {self.current_step_index} の実行中にエラーが発生しました: {e!s}"

    async def _mark_step_completed(self) -> None:
        """現在のステップを完了としてマークします。"""
        if self.current_step_index is None:
            return

        try:
            # ステップを完了としてマーク
            await self.planning_tool.execute(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=self.current_step_index,
                step_status=PlanStepStatus.COMPLETED.value,
            )
            logger.info(
                f"プラン {self.active_plan_id} のステップ {self.current_step_index} を完了としてマークしました"
            )
        except Exception as e:
            logger.warning(f"計画状態の更新に失敗しました: {e}")
            # プランニングツールのストレージで直接ステップ状態を更新
            if self.active_plan_id in self.planning_tool.plans:
                plan_data = self.planning_tool.plans[self.active_plan_id]
                step_statuses = plan_data.get("step_statuses", [])

                # step_statusesリストが十分な長さであることを確認
                while len(step_statuses) <= self.current_step_index:
                    step_statuses.append(PlanStepStatus.NOT_STARTED.value)

                # 状態を更新
                step_statuses[self.current_step_index] = PlanStepStatus.COMPLETED.value
                plan_data["step_statuses"] = step_statuses

    async def _get_plan_text(self) -> str:
        """現在の計画をフォーマットされたテキストとして取得します。"""
        try:
            result = await self.planning_tool.execute(
                command="get", plan_id=self.active_plan_id
            )
            return result.output if hasattr(result, "output") else str(result)
        except Exception as e:
            logger.error(f"計画の取得中にエラーが発生しました: {e}")
            return self._generate_plan_text_from_storage()

    def _generate_plan_text_from_storage(self) -> str:
        """プランニングツールが失敗した場合、ストレージから直接計画テキストを生成します。"""
        try:
            if self.active_plan_id not in self.planning_tool.plans:
                return f"エラー: プランID {self.active_plan_id} が見つかりません"

            plan_data = self.planning_tool.plans[self.active_plan_id]
            title = plan_data.get("title", "無題の計画")
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])
            step_notes = plan_data.get("step_notes", [])

            # step_statusesとstep_notesがステップ数と一致することを確認
            while len(step_statuses) < len(steps):
                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
            while len(step_notes) < len(steps):
                step_notes.append("")

            # 状態別のステップ数をカウント
            status_counts = dict.fromkeys(PlanStepStatus.get_all_statuses(), 0)

            for status in step_statuses:
                if status in status_counts:
                    status_counts[status] += 1

            completed = status_counts[PlanStepStatus.COMPLETED.value]
            total = len(steps)
            progress = (completed / total) * 100 if total > 0 else 0

            plan_text = f"計画: {title} (ID: {self.active_plan_id})\n"
            plan_text += "=" * len(plan_text) + "\n\n"

            plan_text += f"進捗: {completed}/{total} ステップ完了 ({progress:.1f}%)\n"
            plan_text += f"状態: {status_counts[PlanStepStatus.COMPLETED.value]} 完了, {status_counts[PlanStepStatus.IN_PROGRESS.value]} 進行中, "
            plan_text += f"{status_counts[PlanStepStatus.BLOCKED.value]} ブロック中, {status_counts[PlanStepStatus.NOT_STARTED.value]} 未開始\n\n"
            plan_text += "ステップ:\n"

            status_marks = PlanStepStatus.get_status_marks()

            for i, (step, status, notes) in enumerate(
                zip(steps, step_statuses, step_notes, strict=False)
            ):
                # ステップ状態を示すステータスマークを使用
                status_mark = status_marks.get(
                    status, status_marks[PlanStepStatus.NOT_STARTED.value]
                )

                plan_text += f"{i}. {status_mark} {step}\n"
                if notes:
                    plan_text += f"   メモ: {notes}\n"

            return plan_text
        except Exception as e:
            logger.error(
                f"ストレージからの計画テキスト生成中にエラーが発生しました: {e}"
            )
            return f"エラー: プランID {self.active_plan_id} の取得に失敗しました"

    async def _finalize_plan(self) -> str:
        """フローのLLMを直接使用して計画を完了し、要約を提供します。"""
        plan_text = await self._get_plan_text()

        # フローのLLMを直接使用して要約を作成
        try:
            system_message = Message.system_message(
                "あなたは計画立案アシスタントです。完了した計画を要約するのがあなたのタスクです。"
            )

            user_message = Message.user_message(
                f"計画が完了しました。これが最終的な計画状態です:\n\n{plan_text}\n\n達成されたことと最終的な考察の要約を提供してください。"
            )

            response = await self.llm.ask(
                messages=[user_message], system_msgs=[system_message]
            )

            return f"計画完了:\n\n{response}"
        except Exception as e:
            logger.error(f"LLMを使用した計画の完了時にエラーが発生しました: {e}")

            # 要約のためのエージェントへのフォールバック
            try:
                agent = self.primary_agent
                summary_prompt = f"""
                計画が完了しました。これが最終的な計画状態です:

                {plan_text}

                達成されたことと最終的な考察の要約を提供してください。
                """
                summary = await agent.run(summary_prompt)
                return f"計画完了:\n\n{summary}"
            except Exception as e2:
                logger.error(
                    f"エージェントを使用した計画の完了時にエラーが発生しました: {e2}"
                )
                return "計画完了。要約の生成中にエラーが発生しました。"
