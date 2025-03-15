"""タイムアウト付きでシェルコマンドを非同期実行するユーティリティ。"""

import asyncio

TRUNCATED_MESSAGE: str = "<応答が切り取られました><注意>コンテキストを節約するため、このファイルの一部のみが表示されています。`grep -n`でファイル内を検索して探している行番号を見つけてから、このツールを再試行してください。</注意>"
MAX_RESPONSE_LEN: int = 16000


def maybe_truncate(content: str, truncate_after: int | None = MAX_RESPONSE_LEN):
    """コンテンツが指定された長さを超える場合、切り詰めて通知を付加します。"""
    return (
        content
        if not truncate_after or len(content) <= truncate_after
        else content[:truncate_after] + TRUNCATED_MESSAGE
    )


async def run(
    cmd: str,
    timeout: float | None = 120.0,  # seconds
    truncate_after: int | None = MAX_RESPONSE_LEN,
):
    """シェルコマンドをタイムアウト付きで非同期実行します。"""
    process = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return (
            process.returncode or 0,
            maybe_truncate(stdout.decode(), truncate_after=truncate_after),
            maybe_truncate(stderr.decode(), truncate_after=truncate_after),
        )
    except TimeoutError as exc:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        raise TimeoutError(
            f"コマンド '{cmd}' は{timeout}秒後にタイムアウトしました"
        ) from exc
