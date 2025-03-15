from collections import defaultdict
from pathlib import Path
from typing import Literal, get_args

from app.exceptions import ToolError
from app.tool import BaseTool
from app.tool.base import CLIResult, ToolResult
from app.tool.run import run

Command = Literal[
    "view",
    "create",
    "str_replace",
    "insert",
    "undo_edit",
]
SNIPPET_LINES: int = 4

MAX_RESPONSE_LEN: int = 16000

TRUNCATED_MESSAGE: str = "<応答が切り取られました><注意>コンテキストを節約するため、このファイルの一部のみが表示されています。`grep -n`でファイル内を検索して探している行番号を見つけてから、このツールを再試行してください。</注意>"

_STR_REPLACE_EDITOR_DESCRIPTION = """ファイルの表示、作成、編集用のカスタムツール
* 状態はコマンド呼び出しやユーザーとの対話を通じて保持されます
* `path`がファイルの場合、`view`は`cat -n`の結果を表示します。ディレクトリの場合、`view`は2階層までの非隠しファイルとディレクトリを一覧表示します
* 指定された`path`にファイルが既に存在する場合、`create`コマンドは使用できません
* `command`が長い出力を生成する場合、切り詰められて`<応答が切り取られました>`とマークされます
* `undo_edit`コマンドは`path`のファイルに対する最後の編集を元に戻します

`str_replace`コマンドの使用上の注意:
* `old_str`パラメータは元のファイルの1行以上の連続した行と完全に一致する必要があります。空白文字に注意してください！
* `old_str`パラメータがファイル内で一意でない場合、置換は実行されません。`old_str`に十分なコンテキストを含めて一意にしてください
* `new_str`パラメータには、`old_str`を置き換える編集後の行を含める必要があります
"""


def maybe_truncate(content: str, truncate_after: int | None = MAX_RESPONSE_LEN):
    """コンテンツが指定された長さを超える場合、切り詰めて通知を付加します。"""
    return (
        content
        if not truncate_after or len(content) <= truncate_after
        else content[:truncate_after] + TRUNCATED_MESSAGE
    )


class StrReplaceEditor(BaseTool):
    """bashコマンドを実行するためのツール"""

    name: str = "str_replace_editor"
    description: str = _STR_REPLACE_EDITOR_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "description": "実行するコマンド。使用可能なオプション: `view`（表示）, `create`（作成）, `str_replace`（文字列置換）, `insert`（挿入）, `undo_edit`（編集取り消し）",
                "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                "type": "string",
            },
            "path": {
                "description": "ファイルまたはディレクトリの絶対パス。",
                "type": "string",
            },
            "file_text": {
                "description": "`create`コマンドの必須パラメータ。作成するファイルの内容を指定します。",
                "type": "string",
            },
            "old_str": {
                "description": "`str_replace`コマンドの必須パラメータ。置換対象の文字列を指定します。",
                "type": "string",
            },
            "new_str": {
                "description": "`str_replace`コマンドのオプションパラメータ。新しい文字列を指定します（指定がない場合、文字列は追加されません）。`insert`コマンドでは挿入する文字列を指定する必須パラメータになります。",
                "type": "string",
            },
            "insert_line": {
                "description": "`insert`コマンドの必須パラメータ。`new_str`は`path`の`insert_line`行の後に挿入されます。",
                "type": "integer",
            },
            "view_range": {
                "description": "`path`がファイルを指す場合の`view`コマンドのオプションパラメータ。指定がない場合、ファイル全体が表示されます。指定された場合、指定された行番号範囲のファイルが表示されます（例: [11, 12]で11行目と12行目が表示）。インデックスは1から開始します。`[start_line, -1]`を指定すると`start_line`から最後までの行が表示されます。",
                "items": {"type": "integer"},
                "type": "array",
            },
        },
        "required": ["command", "path"],
    }

    _file_history: list = defaultdict(list)

    async def execute(
        self,
        *,
        command: Command,
        path: str,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
        **kwargs,
    ) -> str:
        _path = Path(path)
        self.validate_path(command, _path)
        if command == "view":
            result = await self.view(_path, view_range)
        elif command == "create":
            if file_text is None:
                raise ToolError("createコマンドには`file_text`パラメータが必要です")
            self.write_file(_path, file_text)
            self._file_history[_path].append(file_text)
            result = ToolResult(output=f"ファイルが正常に作成されました: {_path}")
        elif command == "str_replace":
            if old_str is None:
                raise ToolError("str_replaceコマンドには`old_str`パラメータが必要です")
            result = self.str_replace(_path, old_str, new_str)
        elif command == "insert":
            if insert_line is None:
                raise ToolError("insertコマンドには`insert_line`パラメータが必要です")
            if new_str is None:
                raise ToolError("insertコマンドには`new_str`パラメータが必要です")
            result = self.insert(_path, insert_line, new_str)
        elif command == "undo_edit":
            result = self.undo_edit(_path)
        else:
            raise ToolError(
                f"認識できないコマンド {command}。{self.name}ツールで使用可能なコマンド: {', '.join(get_args(Command))}"
            )
        return str(result)

    def validate_path(self, command: str, path: Path):
        """パスとコマンドの組み合わせが有効かチェックします。"""
        # Check if its an absolute path
        if not path.is_absolute():
            suggested_path = Path(path)
            raise ToolError(
                f"パス {path} は絶対パスではありません。'/'で始める必要があります。{suggested_path}の間違いですか？"
            )
        # Check if path exists
        if not path.exists() and command != "create":
            raise ToolError(
                f"パス {path} は存在しません。有効なパスを指定してください。"
            )
        if path.exists() and command == "create":
            raise ToolError(
                f"ファイルが既に存在します: {path}。`create`コマンドでは既存のファイルを上書きできません。"
            )
        # Check if the path points to a directory
        if path.is_dir():
            if command != "view":
                raise ToolError(
                    f"パス {path} はディレクトリで、ディレクトリに対しては`view`コマンドのみ使用できます"
                )

    async def view(self, path: Path, view_range: list[int] | None = None):
        """viewコマンドを実装します"""
        if path.is_dir():
            if view_range:
                raise ToolError(
                    "`path`がディレクトリを指す場合、`view_range`パラメータは使用できません。"
                )

            _, stdout, stderr = await run(
                rf"find {path} -maxdepth 2 -not -path '*/\.*'"
            )
            if not stderr:
                stdout = f"{path}内の2階層までの非隠しファイルとディレクトリ一覧:\n{stdout}\n"
            return CLIResult(output=stdout, error=stderr)

        file_content = self.read_file(path)
        init_line = 1
        if view_range:
            if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                raise ToolError(
                    "無効な`view_range`です。2つの整数からなるリストを指定してください。"
                )
            file_lines = file_content.split("\n")
            n_lines_file = len(file_lines)
            init_line, final_line = view_range
            if init_line < 1 or init_line > n_lines_file:
                raise ToolError(
                    f"無効な`view_range`: {view_range}。最初の要素`{init_line}`はファイルの行範囲内である必要があります: {[1, n_lines_file]}"
                )
            if final_line > n_lines_file:
                raise ToolError(
                    f"無効な`view_range`: {view_range}。2番目の要素`{final_line}`はファイルの行数`{n_lines_file}`より小さい必要があります"
                )
            if final_line != -1 and final_line < init_line:
                raise ToolError(
                    f"無効な`view_range`: {view_range}。2番目の要素`{final_line}`は最初の要素`{init_line}`以上である必要があります"
                )

            if final_line == -1:
                file_content = "\n".join(file_lines[init_line - 1 :])
            else:
                file_content = "\n".join(file_lines[init_line - 1 : final_line])

        return CLIResult(
            output=self._make_output(file_content, str(path), init_line=init_line)
        )

    def str_replace(self, path: Path, old_str: str, new_str: str | None):
        """str_replaceコマンドを実装します。ファイル内容のold_strをnew_strに置換します"""
        # Read the file content
        file_content = self.read_file(path).expandtabs()
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs() if new_str is not None else ""

        # Check if old_str is unique in the file
        occurrences = file_content.count(old_str)
        if occurrences == 0:
            raise ToolError(
                f"置換は実行されませんでした。old_str `{old_str}` は {path} に正確に一致する部分がありませんでした。"
            )
        if occurrences > 1:
            file_content_lines = file_content.split("\n")
            lines = [
                idx + 1
                for idx, line in enumerate(file_content_lines)
                if old_str in line
            ]
            raise ToolError(
                f"置換は実行されませんでした。old_str `{old_str}` が複数の行 {lines} に出現しています。一意になるようにしてください"
            )

        # Replace old_str with new_str
        new_file_content = file_content.replace(old_str, new_str)

        # Write the new content to the file
        self.write_file(path, new_file_content)

        # Save the content to history
        self._file_history[path].append(file_content)

        # Create a snippet of the edited section
        replacement_line = file_content.split(old_str)[0].count("\n")
        start_line = max(0, replacement_line - SNIPPET_LINES)
        end_line = replacement_line + SNIPPET_LINES + new_str.count("\n")
        snippet = "\n".join(new_file_content.split("\n")[start_line : end_line + 1])

        # Prepare the success message
        success_msg = f"ファイル {path} を編集しました。"
        success_msg += self._make_output(
            snippet, f"a snippet of {path}", start_line + 1
        )
        success_msg += "変更内容を確認し、期待通りであることを確認してください。必要に応じてファイルを再編集してください。"

        return CLIResult(output=success_msg)

    def insert(self, path: Path, insert_line: int, new_str: str):
        """insertコマンドを実装します。指定された行にnew_strを挿入します。"""
        file_text = self.read_file(path).expandtabs()
        new_str = new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)

        if insert_line < 0 or insert_line > n_lines_file:
            raise ToolError(
                f"無効な`insert_line`パラメータ: {insert_line}。ファイルの行範囲内である必要があります: {[0, n_lines_file]}"
            )

        new_str_lines = new_str.split("\n")
        new_file_text_lines = (
            file_text_lines[:insert_line]
            + new_str_lines
            + file_text_lines[insert_line:]
        )
        snippet_lines = (
            file_text_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + new_str_lines
            + file_text_lines[insert_line : insert_line + SNIPPET_LINES]
        )

        new_file_text = "\n".join(new_file_text_lines)
        snippet = "\n".join(snippet_lines)

        self.write_file(path, new_file_text)
        self._file_history[path].append(file_text)

        success_msg = f"ファイル {path} を編集しました。"
        success_msg += self._make_output(
            snippet,
            "a snippet of the edited file",
            max(1, insert_line - SNIPPET_LINES + 1),
        )
        success_msg += "変更内容を確認し、期待通り（インデントが正しい、重複行がない等）であることを確認してください。必要に応じてファイルを再編集してください。"
        return CLIResult(output=success_msg)

    def undo_edit(self, path: Path):
        """undo_editコマンドを実装します。"""
        if not self._file_history[path]:
            raise ToolError(f"{path}の編集履歴がありません。")

        old_text = self._file_history[path].pop()
        self.write_file(path, old_text)

        return CLIResult(
            output=f"{path}の最後の編集を元に戻しました。{self._make_output(old_text, str(path))}"
        )

    def read_file(self, path: Path):
        """指定されたパスからファイルの内容を読み込みます。エラーが発生した場合はToolErrorを発生させます。"""
        try:
            return path.read_text()
        except Exception as e:
            raise ToolError(f"{path}の読み込み中にエラーが発生しました: {e}") from None

    def write_file(self, path: Path, file: str):
        """指定されたパスにファイルの内容を書き込みます。エラーが発生した場合はToolErrorを発生させます。"""
        try:
            path.write_text(file)
        except Exception as e:
            raise ToolError(
                f"{path}への書き込み中にエラーが発生しました: {e}"
            ) from None

    def _make_output(
        self,
        file_content: str,
        file_descriptor: str,
        init_line: int = 1,
        expand_tabs: bool = True,
    ):
        """ファイルの内容に基づいてCLI用の出力を生成します。"""
        file_content = maybe_truncate(file_content)
        if expand_tabs:
            file_content = file_content.expandtabs()
        file_content = "\n".join(
            [
                f"{i + init_line:6}\t{line}"
                for i, line in enumerate(file_content.split("\n"))
            ]
        )
        return f"{file_descriptor}に対する`cat -n`の実行結果:\n" + file_content + "\n"
