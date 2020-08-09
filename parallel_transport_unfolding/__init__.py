import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt='[%(asctime)s] %(levelname)s: %(module)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
