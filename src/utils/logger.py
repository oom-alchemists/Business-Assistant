"""日志工具：统一使用 loguru 输出，按天滚动。"""
import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_dir: str = "logs"):
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
    logger.add(log_path / "assistant_{time:YYYY-MM-DD}.log",
               level="DEBUG", rotation="00:00", retention="14 days", encoding="utf-8")
    return logger
