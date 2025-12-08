import logging
import sys

from config.settings import settings


def setup_logger():
    logger = logging.getLogger(settings.APP_NAME)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)8s %(thread)d - %(module)20s.%(funcName)s - %(message)s'
        ))
        logger.addHandler(handler)

    return logger

logger = setup_logger()
