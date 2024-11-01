import sys
import traceback
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """Formats the error message with file name, line number, and error details."""
    # Use traceback to get detailed error information directly
    tb = traceback.extract_tb(error.__traceback__)
    if tb:
        file_name = tb[-1].filename  # The file where the error occurred
        line_number = tb[-1].lineno  # The line number of the error
    else:
        file_name = "<unknown file>"
        line_number = "<unknown line>"

    error_message = (
        f"Error occurred in script: [{file_name}] at line: [{line_number}] "
        f"with error message: [{str(error)}]"
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: Exception):
        super().__init__(error_message)
        # Generate a detailed error message
        self.error_message = error_message_detail(error_detail, error_detail=sys)
        # Log the detailed error message
        logging.error(self.error_message)

    def __str__(self):
        return self.error_message
