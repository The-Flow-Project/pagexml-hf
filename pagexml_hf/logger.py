"""
Logger configuration for the pagexml-hf package with loguru.
"""
from loguru import logger
from pathlib import Path
import sys


def setup_logger(level: str = "DEBUG") -> None:
    """
    Add a debug logger to the root logger.
    """
    logger.remove()

    # Console handler with colored output
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )

    """
    # File handler for all logs with rotation
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        logs_dir / "pagexml-hf.log",
        rotation="5 MB",
        retention="10 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        backtrace=True,
        diagnose=True,
        enqueue=False,  # Thread-safe logging
    )

    # Separate error log file
    logger.add(
        logs_dir / "pagexml-hf_errors.log",
        rotation="5 MB",
        retention="30 days",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        backtrace=True,
        diagnose=True,
        enqueue=False,
    )
    """

    logger.debug(f"Logger initialized with level: {level}")

