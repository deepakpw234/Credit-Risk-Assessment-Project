import os
import sys
import dill

from src.exception import CustomException
from src.logger import logging


def save_object(path,obj):
    try:
        os.makedirs(os.path.join(os.getcwd(),"artifacts","model"),exist_ok=True)

        with open(path,"wb") as file_obj:
            dill.dump(obj,file_obj)


    except Exception as e:
        raise CustomException(e,sys)