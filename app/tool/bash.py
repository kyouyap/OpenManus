import asyncio
import os

from app.exceptions import ToolError
from app.tool.base import BaseTool, CLIResult, ToolResult

_BASH_DESCRIPTION = """ターミナルでbashコマンドを実行します。
* 長時間実行コマンド: 無期限に実行される可能性のあるコマンドは、バックグラウンドで実行し、出力をファイルにリダイレクトする必要があります。例: command = `python3 app.py > server.log 2>&1 &`
* 対話型コマンド: bashコマンドが終了コード`-1`を返した場合、プロセスはまだ終了していないことを意味します。アシスタントは空の`command`でターミナルに2回目の呼び出しを送信して追加のログを取得するか、テキストを送信（`command`にテキストを設定）してSTDINに送信するか、command=`ctrl+c`を送信してプロセスを中断できます。
* タイムアウト: コマンド実行結果が"Command timed out. Sending SIGINT to the process"となった場合、アシスタントはコマンドをバックグラウンドで再実行する必要があります。
"""


class _BashSession:
    """bashシェルのセッション。"""

    _started: bool
    _process: asyncio.subprocess.Process

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self):
        self._started = False
        self._timed_out = False

    async def start(self):
        if self._started:
            return

        self._process = await asyncio.create_subprocess_shell(
            self.command,
            preexec_fn=os.setsid,
            shell=True,
            bufsize=0,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._started = True

    def stop(self):
        """bashシェルを終了します。"""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return
        self._process.terminate()

    async def run(self, command: str):
        """bashシェルでコマンドを実行します。"""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return ToolResult(
                system="tool must be restarted",
                error=f"bash has exited with returncode {self._process.returncode}",
            )
        if self._timed_out:
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            )

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # send command to the process
        self._process.stdin.write(
            command.encode() + f"; echo '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()

        # read output from the process, until the sentinel is found
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    await asyncio.sleep(self._output_delay)
                    # if we read directly from stdout/stderr, it will wait forever for
                    # EOF. use the StreamReader buffer directly instead.
                    output = self._process.stdout._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
                    if self._sentinel in output:
                        # strip the sentinel and break
                        output = output[: output.index(self._sentinel)]
                        break
        except TimeoutError:
            self._timed_out = True
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            ) from None

        output = output.removesuffix("\n")

        error = self._process.stderr._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
        error = error.removesuffix("\n")

        # clear the buffers so that the next output can be read correctly
        self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]

        return CLIResult(output=output, error=error)


class Bash(BaseTool):
    """bashコマンドを実行するためのツール"""

    name: str = "bash"
    description: str = _BASH_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "実行するbashコマンド。前回の終了コードが`-1`の場合は空にして追加のログを表示できます。実行中のプロセスを中断するには`ctrl+c`を使用できます。",
            },
        },
        "required": ["command"],
    }

    _session: _BashSession | None = None

    async def execute(
        self, command: str | None = None, restart: bool = False, **kwargs
    ) -> CLIResult:
        if restart:
            if self._session:
                self._session.stop()
            self._session = _BashSession()
            await self._session.start()

            return ToolResult(system="tool has been restarted.")

        if self._session is None:
            self._session = _BashSession()
            await self._session.start()

        if command is not None:
            return await self._session.run(command)

        raise ToolError("no command provided.")


if __name__ == "__main__":
    bash = Bash()
    rst = asyncio.run(bash.execute("ls -l"))
    print(rst)
