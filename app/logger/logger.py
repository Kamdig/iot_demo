import logging
import os
from datetime import datetime

def initialize_logger(log_dir='logs', log_file=None):

    os.makedirs(log_dir, exist_ok=True)
    if log_file is None:
        log_file = f"system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path, encoding='utf-8'), logging.StreamHandler()]
    )
    logging.info("Logger initialized.")
    logging.info(f"Logging to {log_path}")