import asyncio

from app.agent.manus import Manus
from app.logger import logger


async def main() -> None:
    """エージェントを初期化し、プロンプトに応答します。"""
    agent = Manus()
    try:
        prompt = input("プロンプトを入力してください: ")
        if not prompt.strip():
            logger.warning("空のプロンプトが入力されました。")
            return

        logger.warning("リクエストを処理しています...")
        await agent.run(prompt)
        logger.info("リクエスト処理が完了しました。")
    except KeyboardInterrupt:
        logger.warning("操作が中断されました。")


if __name__ == "__main__":
    asyncio.run(main())
