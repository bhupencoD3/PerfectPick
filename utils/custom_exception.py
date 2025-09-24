import sys
import logging
from utils.logger import get_logger

logger = get_logger(__name__)

class CustomException(Exception):
    """
    Base class for custom exceptions in the project.
    """

    def __init__(self, message: str, error_detail: sys = None):
        super().__init__(message)
        self.message = message
        self.error_detail = error_detail
        self.log_error()

    def log_error(self):
        if self.error_detail:
            _, _, exc_tb = self.error_detail.exc_info()
            file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown"
            line_no = exc_tb.tb_lineno if exc_tb else "Unknown"
            logger.error(
                f"Exception: {self.message} | File: {file_name} | Line: {line_no}"
            )
        else:
            logger.error(f"Exception: {self.message}")

    def __str__(self):
        return self.message
