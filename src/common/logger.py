import logging

PROJECT_NAME = "pytorch-lightning-template"

logger = logging.getLogger(PROJECT_NAME)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    logging.Formatter(
        f"[%(asctime)s] [{PROJECT_NAME}] %(levelname)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
)
logging.basicConfig(level=logging.INFO, handlers=[stream_handler], force=True)
