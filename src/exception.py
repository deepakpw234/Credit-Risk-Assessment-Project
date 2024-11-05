import os
import sys
from src.logger import logging

def get_error_details(error,error_details:sys):
    _,_,exc_tab = error_details.exc_info()
    file_name = exc_tab.tb_frame.f_code.co_filename    
    error_message = f"The python error occured in the file [{file_name}] at line number [{exc_tab.tb_lineno}] with the error message [{str(error)}]"

    return error_message

class CustomException(Exception):
    def __init__(self, error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message = get_error_details(error_message,error_details)

    def __str__(self):
        return self.error_message
    


