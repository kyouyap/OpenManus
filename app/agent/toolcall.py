import json
from typing import Any

from pydantic import Field

from app.agent.react import ReActAgent
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import TOOL_CHOICE_TYPE, AgentState, Message, ToolCall, ToolChoice
from app.tool import CreateChatCompletion, Terminate, ToolCollection

TOOL_CALL_REQUIRED = "ツール呼び出しが必要ですが、提供されていません"


class ToolCallAgent(ReActAgent):
    """ツール/関数呼び出しを強化された抽象化で処理するための基本エージェントクラス"""

    name: str = "toolcall"
    description: str = "ツール呼び出しを実行できるエージェント。"

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: list[ToolCall] = Field(default_factory=list)

    max_steps: int = 30
    max_observe: int | bool | None = None

    async def think(self) -> bool:
        """現在の状態を処理し、ツールを使用して次のアクションを決定します"""
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        try:
            # ツールオプション付きのレスポンスを取得
            response = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=[Message.system_message(self.system_prompt)]
                if self.system_prompt
                else None,
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )
        except ValueError:
            raise
        except Exception as e:
            # RetryErrorにTokenLimitExceededが含まれているかチェック
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(
                    f"🚨 トークン制限エラー (RetryErrorより): {token_limit_error}"
                )
                self.memory.add_message(
                    Message.assistant_message(
                        f"最大トークン制限に達したため、実行を継続できません: {token_limit_error!s}"
                    )
                )
                self.state = AgentState.FINISHED
                return False
            raise

        self.tool_calls = response.tool_calls

        # レスポンス情報をログに記録
        logger.info(f"✨ {self.name}の思考: {response.content}")
        logger.info(
            f"🛠️ {self.name}は{len(response.tool_calls) if response.tool_calls else 0}個のツールを使用することを選択しました"
        )
        if response.tool_calls:
            logger.info(
                f"🧰 準備中のツール: {[call.function.name for call in response.tool_calls]}"
            )

        try:
            # 異なるtool_choicesモードを処理
            if self.tool_choices == ToolChoice.NONE:
                if response.tool_calls:
                    logger.warning(
                        f"🤔 {self.name}は利用できないツールを使おうとしました！"
                    )
                if response.content:
                    self.memory.add_message(Message.assistant_message(response.content))
                    return True
                return False

            # アシスタントメッセージを作成して追加
            assistant_msg = (
                Message.from_tool_calls(
                    content=response.content, tool_calls=self.tool_calls
                )
                if self.tool_calls
                else Message.assistant_message(response.content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # act()で処理される

            # 'auto'モードの場合、コマンドがなくてもコンテンツがあれば続行
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(response.content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 {self.name}の思考プロセスでエラーが発生しました: {e}")
            self.memory.add_message(
                Message.assistant_message(f"処理中にエラーが発生しました: {e!s}")
            )
            return False

    async def act(self) -> str:
        """ツール呼び出しを実行し、その結果を処理します"""
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)

            # ツール呼び出しがない場合は最後のメッセージ内容を返す
            return (
                self.messages[-1].content
                or "実行するコンテンツまたはコマンドがありません"
            )

        results = []
        for command in self.tool_calls:
            result = await self.execute_tool(command)

            if self.max_observe:
                result = result[: self.max_observe]

            logger.info(
                f"🎯 ツール '{command.function.name}' が完了しました！ 結果: {result}"
            )

            # ツールのレスポンスをメモリに追加
            tool_msg = Message.tool_message(
                content=result, tool_call_id=command.id, name=command.function.name
            )
            self.memory.add_message(tool_msg)
            results.append(result)

        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """堅牢なエラー処理を備えた単一のツール呼び出しを実行します"""
        if not command or not command.function or not command.function.name:
            return "エラー: 無効なコマンド形式"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"エラー: 不明なツール '{name}'"

        try:
            # 引数をパース
            args = json.loads(command.function.arguments or "{}")

            # ツールを実行
            logger.info(f"🔧 ツールを起動中: '{name}'...")
            result = await self.available_tools.execute(name=name, tool_input=args)

            # 表示用に結果をフォーマット
            observation = (
                f"コマンド `{name}` の実行結果:\n{result!s}"
                if result
                else f"コマンド `{name}` は出力なしで完了しました"
            )

            # `finish`などの特殊ツールを処理
            await self._handle_special_tool(name=name, result=result)

            return observation
        except json.JSONDecodeError:
            error_msg = f"{name}の引数のパースエラー: 無効なJSON形式"
            logger.error(
                f"📝 '{name}'の引数が不正です - 無効なJSON、引数:{command.function.arguments}"
            )
            return f"エラー: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ ツール '{name}' でエラーが発生しました: {e!s}"
            logger.error(error_msg)
            return f"エラー: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """特殊ツールの実行と状態変更を処理します"""
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # エージェントの状態を完了に設定
            logger.info(f"🏁 特殊ツール '{name}' がタスクを完了しました！")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """ツールの実行によってエージェントを終了すべきかを判断します"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """ツール名が特殊ツールリストに含まれているかをチェックします"""
        return name.lower() in [n.lower() for n in self.special_tool_names]
