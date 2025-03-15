from pathlib import Path
from typing import ClassVar, NotRequired, TypedDict

import aiofiles

from app.tool.base import BaseTool


class FileParams(TypedDict):
    """ファイル保存ツールのパラメータ。"""

    content: str
    file_path: str
    mode: NotRequired[str]


class FileSaver(BaseTool):
    """指定されたパスにローカルファイルとしてコンテンツを保存します。"""

    name: str = "file_saver"
    description: str = """指定されたパスにローカルファイルとしてコンテンツを保存します。
テキスト、コード、生成されたコンテンツをローカルファイルシステムに保存する必要がある場合にこのツールを使用します。
このツールはコンテンツとファイルパスを受け取り、指定された場所にコンテンツを保存します。
"""

    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "(必須) ファイルに保存するコンテンツ。",
            },
            "file_path": {
                "type": "string",
                "description": (
                    "(必須) ファイル名と拡張子を含む、ファイルを保存するパス。"
                ),
            },
            "mode": {
                "type": "string",
                "description": (
                    "(オプション) ファイルを開くモード。"
                    "デフォルトは書き込み用の'w'。"
                    "追記用には'a'を使用。"
                ),
                "enum": ["w", "a"],
                "default": "w",
            },
        },
        "required": ["content", "file_path"],
    }

    async def execute(self, **kwargs: FileParams) -> str:
        """指定されたパスにコンテンツをファイルとして保存します。"""
        try:
            content: str = str(kwargs["content"])
            file_path_str: str = str(kwargs["file_path"])
            mode_str: str = str(kwargs.get("mode", "w"))
            file_path = Path(file_path_str)

            # ディレクトリが存在することを確認
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)

            # ファイルに直接書き込み
            async with aiofiles.open(
                str(file_path), mode="w" if mode_str == "w" else "a", encoding="utf-8"
            ) as file:
                await file.write(content)
        except OSError as e:
            return f"ファイルの保存中にエラーが発生しました: {e!s}"
        except KeyError as e:
            return f"必要なパラメータが不足しています: {e!s}"
        else:
            return f"コンテンツを {file_path} に保存しました"
