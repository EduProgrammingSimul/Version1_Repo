import logging
from logging import Logger
from pathlib import Path
import os

_LOGGER_CREATED = False


def get_log_file_path() -> str:

    root = Path(__file__).resolve().parent.parent
    log_file = os.environ.get("PIPELINE_LOG_FILE", "pipeline.log")
    return str((root / log_file).resolve())


def get_logger(name: str = "pipeline") -> Logger:
    global _LOGGER_CREATED
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not _LOGGER_CREATED:
        log_path = get_log_file_path()
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(module)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        if not any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == fh.baseFilename
            for h in root_logger.handlers
        ):
            root_logger.addHandler(fh)
        root_logger.debug(f"[logging_setup] Central log initialized at {log_path}")
        _LOGGER_CREATED = True
    return logger
