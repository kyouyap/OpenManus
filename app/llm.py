import tiktoken
from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.config import LLMSettings, config
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)

REASONING_MODELS = ["o1", "o3-mini"]


class LLM:
    _instances: dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: LLMSettings | None = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: LLMSettings | None = None
    ):
        if not hasattr(self, "client"):  # 初期化済みでない場合のみ初期化
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])
            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.api_key = llm_config.api_key
            self.api_version = llm_config.api_version
            self.base_url = llm_config.base_url

            # トークン計算関連の属性を追加
            self.total_input_tokens = 0
            self.max_input_tokens = (
                llm_config.max_input_tokens
                if hasattr(llm_config, "max_input_tokens")
                else None
            )

            # トークナイザーの初期化
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # モデルがtiktokenのプリセットに存在しない場合、デフォルトとしてcl100k_baseを使用
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

            if self.api_type == "azure":
                self.client = AsyncAzureOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
            else:
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def count_tokens(self, text: str) -> int:
        """テキスト内のトークン数を計算します"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: list[dict]) -> int:
        """メッセージリスト内のトークン数を計算します"""
        token_count = 0
        for message in messages:
            # 各メッセージの基本トークン数（OpenAIの計算方法に従う）
            token_count += 4  # 各メッセージの基本トークン数

            # ロールのトークン数を計算
            if "role" in message:
                token_count += self.count_tokens(message["role"])

            # コンテンツのトークン数を計算
            if message.get("content"):
                token_count += self.count_tokens(message["content"])

            # ツール呼び出しのトークン数を計算
            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    if "function" in tool_call:
                        # 関数名
                        if "name" in tool_call["function"]:
                            token_count += self.count_tokens(
                                tool_call["function"]["name"]
                            )
                        # 関数の引数
                        if "arguments" in tool_call["function"]:
                            token_count += self.count_tokens(
                                tool_call["function"]["arguments"]
                            )

            # ツールレスポンスのトークン数を計算
            if message.get("name"):
                token_count += self.count_tokens(message["name"])

            if message.get("tool_call_id"):
                token_count += self.count_tokens(message["tool_call_id"])

        # メッセージフォーマットの追加トークン
        token_count += 2  # メッセージフォーマットの追加トークン

        return token_count

    def update_token_count(self, input_tokens: int) -> None:
        """トークン数を更新します"""
        # max_input_tokensが設定されている場合のみトークンを追跡
        self.total_input_tokens += input_tokens
        logger.info(
            f"トークン使用量: 入力={input_tokens}, 累積入力={self.total_input_tokens}"
        )

    def check_token_limit(self, input_tokens: int) -> bool:
        """トークン制限を超えていないかチェックします"""
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        # max_input_tokensが設定されていない場合は常にTrue
        return True

    def get_limit_error_message(self, input_tokens: int) -> str:
        """トークン制限超過のエラーメッセージを生成します"""
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            return f"リクエストが入力トークン制限を超える可能性があります（現在: {self.total_input_tokens}, 必要: {input_tokens}, 最大: {self.max_input_tokens}）"

        return "トークン制限を超過しました"

    @staticmethod
    def format_messages(messages: list[dict | Message]) -> list[dict]:
        """メッセージをLLM用にフォーマットし、OpenAIメッセージ形式に変換します。

        引数:
            messages: dictまたはMessageオブジェクトのメッセージリスト

        戻り値:
            List[dict]: OpenAI形式でフォーマットされたメッセージのリスト

        例外:
            ValueError: メッセージが無効または必須フィールドが欠けている場合
            TypeError: サポートされていないメッセージタイプが提供された場合

        例:
            >>> msgs = [
            ...     Message.system_message("あなたは役立つアシスタントです"),
            ...     {"role": "user", "content": "こんにちは"},
            ...     Message.user_message("調子はどうですか？")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            if isinstance(message, Message):
                message = message.to_dict()
            if isinstance(message, dict):
                # メッセージがdictの場合、必須フィールドを確認
                if "role" not in message:
                    raise ValueError("メッセージdictには'role'フィールドが必要です")
                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
                # それ以外の場合はメッセージを含めない
            else:
                raise TypeError(
                    f"サポートされていないメッセージタイプ: {type(message)}"
                )

        # 全てのメッセージが必須フィールドを持っているか検証
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"無効なロール: {msg['role']}")

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # TokenLimitExceededは再試行しない
    )
    async def ask(
        self,
        messages: list[dict | Message],
        system_msgs: list[dict | Message] | None = None,
        stream: bool = True,
        temperature: float | None = None,
    ) -> str:
        """LLMにプロンプトを送信して応答を取得します。

        引数:
            messages: 会話メッセージのリスト
            system_msgs: 先頭に追加するオプションのシステムメッセージ
            stream (bool): 応答をストリーミングするかどうか
            temperature (float): 応答のサンプリング温度

        戻り値:
            str: 生成された応答

        例外:
            TokenLimitExceeded: トークン制限を超えた場合
            ValueError: メッセージが無効または応答が空の場合
            OpenAIError: APIコールが再試行後も失敗した場合
            Exception: 予期しないエラーの場合
        """
        try:
            # システムメッセージとユーザーメッセージをフォーマット
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # 入力トークン数を計算
            input_tokens = self.count_message_tokens(messages)

            # トークン制限を超えていないかチェック
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # 再試行されない特別な例外を発生
                raise TokenLimitExceeded(error_message)

            params = {
                "model": self.model,
                "messages": messages,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            if not stream:
                # 非ストリーミングリクエスト
                params["stream"] = False

                response = await self.client.chat.completions.create(**params)

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("LLMからの応答が空または無効です")

                # トークン数を更新
                self.update_token_count(response.usage.prompt_tokens)

                return response.choices[0].message.content

            # ストリーミングリクエスト、リクエスト前に推定トークン数を更新
            self.update_token_count(input_tokens)

            params["stream"] = True
            response = await self.client.chat.completions.create(**params)

            collected_messages = []
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                print(chunk_message, end="", flush=True)

            print()  # ストリーミング後の改行
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("ストリーミングLLMからの応答が空です")

            return full_response

        except TokenLimitExceeded:
            # トークン制限エラーはログを記録せずに再発生
            raise
        except ValueError as ve:
            logger.error(f"バリデーションエラー: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI APIエラー: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("認証に失敗しました。APIキーを確認してください。")
            elif isinstance(oe, RateLimitError):
                logger.error(
                    "レート制限を超過しました。再試行回数の増加を検討してください。"
                )
            elif isinstance(oe, APIError):
                logger.error(f"APIエラー: {oe}")
            raise
        except Exception as e:
            logger.error(f"askでの予期しないエラー: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # TokenLimitExceededは再試行しない
    )
    async def ask_tool(
        self,
        messages: list[dict | Message],
        system_msgs: list[dict | Message] | None = None,
        timeout: int = 300,
        tools: list[dict] | None = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: float | None = None,
        **kwargs,
    ):
        """関数/ツールを使用してLLMに問い合わせ、応答を返します。

        引数:
            messages: 会話メッセージのリスト
            system_msgs: 先頭に追加するオプションのシステムメッセージ
            timeout: リクエストのタイムアウト（秒）
            tools: 使用するツールのリスト
            tool_choice: ツール選択戦略
            temperature: 応答のサンプリング温度
            **kwargs: 追加の補完引数

        戻り値:
            ChatCompletionMessage: モデルの応答

        例外:
            TokenLimitExceeded: トークン制限を超えた場合
            ValueError: ツール、tool_choice、またはメッセージが無効な場合
            OpenAIError: APIコールが再試行後も失敗した場合
            Exception: 予期しないエラーの場合
        """
        try:
            # tool_choiceを検証
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"無効なtool_choice: {tool_choice}")

            # メッセージをフォーマット
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # 入力トークン数を計算
            input_tokens = self.count_message_tokens(messages)

            # ツールがある場合、ツール説明のトークン数を計算
            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))

            input_tokens += tools_tokens

            # トークン制限を超えていないかチェック
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # 再試行されない特別な例外を発生
                raise TokenLimitExceeded(error_message)

            # ツールが提供された場合は検証
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError(
                            "各ツールは'type'フィールドを持つdictでなければなりません"
                        )

            # 補完リクエストの設定
            params = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                **kwargs,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            response = await self.client.chat.completions.create(**params)

            # 応答が有効か確認
            if not response.choices or not response.choices[0].message:
                print(response)
                raise ValueError("LLMからの応答が無効または空です")

            # トークン数を更新
            self.update_token_count(response.usage.prompt_tokens)

            return response.choices[0].message

        except TokenLimitExceeded:
            # トークン制限エラーはログを記録せずに再発生
            raise
        except ValueError as ve:
            logger.error(f"ask_toolでのバリデーションエラー: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI APIエラー: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("認証に失敗しました。APIキーを確認してください。")
            elif isinstance(oe, RateLimitError):
                logger.error(
                    "レート制限を超過しました。再試行回数の増加を検討してください。"
                )
            elif isinstance(oe, APIError):
                logger.error(f"APIエラー: {oe}")
            raise
        except Exception as e:
            logger.error(f"ask_toolでの予期しないエラー: {e}")
            raise
