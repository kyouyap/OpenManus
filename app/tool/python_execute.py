import multiprocessing
import sys
from io import StringIO

from app.tool.base import BaseTool


class PythonExecute(BaseTool):
    """タイムアウトと安全性制限付きでPythonコードを実行するツール。"""

    name: str = "python_execute"
    description: str = "Pythonコードを実行します。注意: print出力のみが表示され、関数の戻り値は取得されません。結果を確認するにはprint文を使用してください。"
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "実行するPythonコード。",
            },
        },
        "required": ["code"],
    }

    def _run_code(self, code: str, result_dict: dict, safe_globals: dict) -> None:
        original_stdout = sys.stdout
        try:
            output_buffer = StringIO()
            sys.stdout = output_buffer
            exec(code, safe_globals, safe_globals)
            result_dict["observation"] = output_buffer.getvalue()
            result_dict["success"] = True
        except Exception as e:
            result_dict["observation"] = str(e)
            result_dict["success"] = False
        finally:
            sys.stdout = original_stdout

    async def execute(
        self,
        code: str,
        timeout: int = 5,
    ) -> dict:
        """指定されたPythonコードをタイムアウト付きで実行します。

        引数:
            code (str): 実行するPythonコード
            timeout (int): 実行タイムアウト（秒）

        戻り値:
            Dict: 実行出力またはエラーメッセージを含む'output'と'success'ステータスを含む辞書
        """
        with multiprocessing.Manager() as manager:
            result = manager.dict({"observation": "", "success": False})
            if isinstance(__builtins__, dict):
                safe_globals = {"__builtins__": __builtins__}
            else:
                safe_globals = {"__builtins__": __builtins__.__dict__.copy()}
            proc = multiprocessing.Process(
                target=self._run_code, args=(code, result, safe_globals)
            )
            proc.start()
            proc.join(timeout)

            # timeout process
            if proc.is_alive():
                proc.terminate()
                proc.join(1)
                return {
                    "observation": f"{timeout}秒後に実行がタイムアウトしました",
                    "success": False,
                }
            return dict(result)
