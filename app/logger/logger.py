from datetime import datetime
import logging
import os

def initialize_logger(log_dir='logs', log_file=None):
    os.makedirs(log_dir, exist_ok=True)
    if log_file is None:
        log_file = f"system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_file)

    # Read log level from environment, default to INFO
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logger initialized at level {log_level}.")
    logging.info(f"Logging to {log_path}")