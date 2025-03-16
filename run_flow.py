import asyncio
import time

from app.agent.manus import Manus
from app.flow.base import FlowType
from app.flow.flow_factory import FlowFactory
from app.logger import logger


async def run_flow():
    """フローを実行するメイン関数"""
    agents = {
        "manus": Manus(),
    }

    try:
        prompt = input("プロンプトを入力してください: ")

        if prompt.strip().isspace() or not prompt:
            logger.warning("空のプロンプトが入力されました。")
            return

        flow = FlowFactory.create_flow(
            flow_type=FlowType.PLANNING,
            agents=agents,
        )
        logger.warning("リクエストを処理中...")

        try:
            start_time = time.time()
            result = await asyncio.wait_for(
                flow.execute(prompt),
                timeout=3600,  # 実行全体のタイムアウトを60分に設定
            )
            elapsed_time = time.time() - start_time
            logger.info(f"リクエストの処理が {elapsed_time:.2f} 秒で完了しました")
            logger.info(result)
        except TimeoutError:
            logger.error("リクエストの処理が1時間でタイムアウトしました")
            logger.info(
                "タイムアウトにより処理が終了しました。より簡単なリクエストを試してください。"
            )

    except KeyboardInterrupt:
        logger.info("ユーザーによって処理がキャンセルされました。")
    except Exception as e:
        logger.error(f"エラー: {e!s}")


if __name__ == "__main__":
    asyncio.run(run_flow())
