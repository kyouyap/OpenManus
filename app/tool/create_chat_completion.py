from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, Field

from app.tool import BaseTool


class CreateChatCompletion(BaseTool):
    name: str = "create_chat_completion"
    description: str = "指定された出力形式で構造化された補完を生成します。"

    # Type mapping for JSON schema
    type_mapping: dict = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
    }
    response_type: type | None = None
    required: list[str] = Field(default_factory=lambda: ["response"])

    def __init__(self, response_type: type | None = str):
        """特定のレスポンス型で初期化します。"""
        super().__init__()
        self.response_type = response_type
        self.parameters = self._build_parameters()

    def _build_parameters(self) -> dict:
        """レスポンス型に基づいてパラメータスキーマを構築します。"""
        if self.response_type == str:
            return {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "ユーザーに配信されるべきレスポンステキスト。",
                    },
                },
                "required": self.required,
            }

        if isinstance(self.response_type, type) and issubclass(
            self.response_type, BaseModel
        ):
            schema = self.response_type.model_json_schema()
            return {
                "type": "object",
                "properties": schema["properties"],
                "required": schema.get("required", self.required),
            }

        return self._create_type_schema(self.response_type)

    def _create_type_schema(self, type_hint: type) -> dict:
        """指定された型のJSONスキーマを作成します。"""
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        # プリミティブ型の処理
        if origin is None:
            return {
                "type": "object",
                "properties": {
                    "response": {
                        "type": self.type_mapping.get(type_hint, "string"),
                        "description": f"Response of type {type_hint.__name__}",
                    }
                },
                "required": self.required,
            }

        # リスト型の処理
        if origin is list:
            item_type = args[0] if args else Any
            return {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "array",
                        "items": self._get_type_info(item_type),
                    }
                },
                "required": self.required,
            }

        # 辞書型の処理
        if origin is dict:
            value_type = args[1] if len(args) > 1 else Any
            return {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "object",
                        "additionalProperties": self._get_type_info(value_type),
                    }
                },
                "required": self.required,
            }

        # Union型の処理
        if origin is Union:
            return self._create_union_schema(args)

        return self._build_parameters()

    def _get_type_info(self, type_hint: type) -> dict:
        """単一の型の型情報を取得します。"""
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            return type_hint.model_json_schema()

        return {
            "type": self.type_mapping.get(type_hint, "string"),
            "description": f"型{getattr(type_hint, '__name__', 'any')}の値",
        }

    def _create_union_schema(self, types: tuple) -> dict:
        """Union型のスキーマを作成します。"""
        return {
            "type": "object",
            "properties": {
                "response": {"anyOf": [self._get_type_info(t) for t in types]}
            },
            "required": self.required,
        }

    async def execute(self, required: list | None = None, **kwargs) -> Any:
        """型変換を伴うチャット補完を実行します。

        引数:
            required: 必須フィールド名のリストまたはNone
            **kwargs: レスポンスデータ

        戻り値:
            response_typeに基づいて変換されたレスポンス
        """
        required = required or self.required

        # requiredがリストの場合の処理
        if isinstance(required, list) and len(required) > 0:
            if len(required) == 1:
                required_field = required[0]
                result = kwargs.get(required_field, "")
            else:
                # 複数のフィールドを辞書として返す
                return {field: kwargs.get(field, "") for field in required}
        else:
            required_field = "response"
            result = kwargs.get(required_field, "")

        # 型変換ロジック
        if self.response_type == str:
            return result

        if isinstance(self.response_type, type) and issubclass(
            self.response_type, BaseModel
        ):
            return self.response_type(**kwargs)

        if get_origin(self.response_type) in (list, dict):
            return result  # Assuming result is already in correct format

        try:
            return self.response_type(result)
        except (ValueError, TypeError):
            return result
