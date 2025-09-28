import logging
import json
from logging.handlers import QueueHandler, Queue
import queue
from datetime import datetime

def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Returns a logger instance configured with a thread-safe QueueHandler for cloud environments.
    Outputs JSON-formatted logs to stdout for cloud monitoring.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Thread-safe queue for concurrent logging
        log_queue = queue.Queue(-1)  # No size limit
        queue_handler = QueueHandler(log_queue)
        logger.addHandler(queue_handler)

        # Console handler (stdout for cloud logging)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}',
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        
        # Queue listener to process logs from queue to console
        queue_listener = logging.handlers.QueueListener(log_queue, console_handler)
        queue_listener.start()
        
        # Ensure listener stops cleanly on shutdown
        import atexit
        atexit.register(queue_listener.stop)

    return logger