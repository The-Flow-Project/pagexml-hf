import loguru
import sys

logger = loguru.logger
logger.remove()

def init_debug_logger():
    """
    Add a debug logger to the root logger.
    """
    logger.add(
        sys.stdout,
        colorize=True,
        level="DEBUG",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <lvl>{level: <8}</lvl> | {message}",
    )
    logger.info("Logger initialized")
    return logger

def init_info_logger():
    """
    Add a info logger to the root logger.
    """
    logger.add(
        sys.stdout,
        colorize=True,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <lvl>{level: <8}</lvl> | {message}",
    )
    logger.info("Logger initialized")
    return logger