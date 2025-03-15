import asyncio
import os
import shlex

from app.tool.base import BaseTool, CLIResult


class Terminal(BaseTool):
    name: str = "execute_command"
    description: str = """システムでCLIコマンドを実行するためのリクエストです。
システム操作を実行したり、ユーザーのタスクの任意のステップを実行するための特定のコマンドを実行する必要がある場合に使用します。
コマンドはユーザーのシステムに合わせて調整し、何を行うのかを明確に説明する必要があります。
実行可能なスクリプトを作成するよりも、複雑なCLIコマンドの実行を優先してください。より柔軟で実行が容易だからです。
コマンドは現在の作業ディレクトリで実行されます。
注意: 50ms未満で完了するコマンドについては、コマンドの最後に`sleep 0.05`を追加する必要があります。これは、コマンドが早すぎると出力が返されないことがあるターミナルツールの既知の問題を回避するためです。
"""
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "（必須）実行するCLIコマンド。現在のオペレーティングシステムで有効である必要があります。コマンドが適切にフォーマットされており、有害な指示を含んでいないことを確認してください。",
            }
        },
        "required": ["command"],
    }
    process: asyncio.subprocess.Process | None = None
    current_path: str = os.getcwd()
    lock: asyncio.Lock = asyncio.Lock()

    async def execute(self, command: str) -> CLIResult:
        """永続的なコンテキストで非同期にターミナルコマンドを実行します。

        引数:
            command (str): 実行するターミナルコマンド

        戻り値:
            str: コマンド実行の出力とエラー
        """
        # Split the command by & to handle multiple commands
        commands = [cmd.strip() for cmd in command.split("&") if cmd.strip()]
        final_output = CLIResult(output="", error="")

        for cmd in commands:
            sanitized_command = self._sanitize_command(cmd)

            # Handle 'cd' command internally
            if sanitized_command.lstrip().startswith("cd "):
                result = await self._handle_cd_command(sanitized_command)
            else:
                async with self.lock:
                    try:
                        self.process = await asyncio.create_subprocess_shell(
                            sanitized_command,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            cwd=self.current_path,
                        )
                        stdout, stderr = await self.process.communicate()
                        result = CLIResult(
                            output=stdout.decode().strip(),
                            error=stderr.decode().strip(),
                        )
                    except Exception as e:
                        result = CLIResult(output="", error=str(e))
                    finally:
                        self.process = None

            # Combine outputs
            if result.output:
                final_output.output += (
                    (result.output + "\n") if final_output.output else result.output
                )
            if result.error:
                final_output.error += (
                    (result.error + "\n") if final_output.error else result.error
                )

        # Remove trailing newlines
        final_output.output = final_output.output.rstrip()
        final_output.error = final_output.error.rstrip()
        return final_output

    async def execute_in_env(self, env_name: str, command: str) -> CLIResult:
        """指定されたConda環境内で非同期にターミナルコマンドを実行します。

        引数:
            env_name (str): Conda環境の名前
            command (str): 環境内で実行するターミナルコマンド

        戻り値:
            str: コマンド実行の出力とエラー
        """
        sanitized_command = self._sanitize_command(command)

        # Construct the command to run within the Conda environment
        # Using 'conda run -n env_name command' to execute without activating
        conda_command = f"conda run -n {shlex.quote(env_name)} {sanitized_command}"

        return await self.execute(conda_command)

    async def _handle_cd_command(self, command: str) -> CLIResult:
        """現在のパスを変更するための'cd'コマンドを処理します。

        引数:
            command (str): 処理する'cd'コマンド

        戻り値:
            TerminalOutput: 'cd'コマンドの結果
        """
        try:
            parts = shlex.split(command)
            if len(parts) < 2:
                new_path = os.path.expanduser("~")
            else:
                new_path = os.path.expanduser(parts[1])

            # Handle relative paths
            if not os.path.isabs(new_path):
                new_path = os.path.join(self.current_path, new_path)

            new_path = os.path.abspath(new_path)

            if os.path.isdir(new_path):
                self.current_path = new_path
                return CLIResult(
                    output=f"ディレクトリを{self.current_path}に変更しました", error=""
                )
            return CLIResult(output="", error=f"ディレクトリが存在しません: {new_path}")
        except Exception as e:
            return CLIResult(output="", error=str(e))

    @staticmethod
    def _sanitize_command(command: str) -> str:
        """安全な実行のためにコマンドをサニタイズします。

        引数:
            command (str): サニタイズするコマンド

        戻り値:
            str: サニタイズされたコマンド
        """
        # Example sanitization: restrict certain dangerous commands
        dangerous_commands = ["rm", "sudo", "shutdown", "reboot"]
        try:
            parts = shlex.split(command)
            if any(cmd in dangerous_commands for cmd in parts):
                raise ValueError("危険なコマンドの使用は制限されています。")
        except Exception:
            # If shlex.split fails, try basic string comparison
            if any(cmd in command for cmd in dangerous_commands):
                raise ValueError("Use of dangerous commands is restricted.")

        # Additional sanitization logic can be added here
        return command

    async def close(self):
        """永続的なシェルプロセスが存在する場合、それを終了します。"""
        async with self.lock:
            if self.process:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5)
                except TimeoutError:
                    self.process.kill()
                    await self.process.wait()
                finally:
                    self.process = None

    async def __aenter__(self):
        """非同期コンテキストマネージャーを開始します。"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャーを終了し、プロセスを閉じます。"""
        await self.close()
