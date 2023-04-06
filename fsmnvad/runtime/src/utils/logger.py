# -*- coding:utf-8 -*-
# @FileName  :logger.py
# @Time      :2023/4/3 18:18
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import logging
from datetime import datetime


def setup_logger(
    log_level: str = "info",
) -> None:
    """Setup log level.

    Args:
      log_level:
        The log level to use, e.g., "debug", "info", "warning", "error",
        "critical"
    """

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    level = logging.ERROR

    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "warning":
        level = logging.WARNING
    elif log_level == "critical":
        level = logging.CRITICAL

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(formatter))
    logging.getLogger("").addHandler(console)
