import sys
from datetime import datetime

from loguru import logger as _logger

from app.config import PROJECT_ROOT

_print_level = "INFO"


def define_log_level(print_level="INFO", logfile_level="DEBUG", name: str = None):
    """ログレベルを指定されたレベルに調整します"""
    global _print_level
    _print_level = print_level

    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d%H%M%S")
    log_name = (
        f"{name}_{formatted_date}" if name else formatted_date
    )  # 接頭辞付きのログ名を生成

    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    _logger.add(PROJECT_ROOT / f"logs/{log_name}.log", level=logfile_level)
    return _logger


logger = define_log_level()

if __name__ == "__main__":
    logger.info("アプリケーションを開始しています")
    logger.debug("デバッグメッセージ")
    logger.warning("警告メッセージ")
    logger.error("エラーメッセージ")
    logger.critical("重大なメッセージ")

    try:
        raise ValueError("テストエラー")
    except Exception as e:
        logger.exception(f"エラーが発生しました: {e}")
