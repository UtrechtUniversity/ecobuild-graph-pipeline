import logging
from logging.handlers import RotatingFileHandler
import os
from .general_config import LOG_LEVEL

# Create log dir if needed
os.makedirs("logs", exist_ok=True)

# Formatter
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

# Stream handler (stdout — for Docker logs)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# File handler (optional: log file inside container)
file_handler = RotatingFileHandler("logs/app.log", maxBytes=5_000_000, backupCount=3)
file_handler.setFormatter(formatter)

# Root logger setup
logging.basicConfig(
    level=LOG_LEVEL,
    handlers=[stream_handler, file_handler]
)

logger = logging.getLogger("Knowledge Extractor")


