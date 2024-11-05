import os
import sys
import urllib.request
import pandas as pd
import numpy as np
import urllib
import zipfile

from dataclasses import dataclass


from src.exception import CustomException
from src.logger import logging



@dataclass
class DataIngestionConfig:
    zip_dataset_path = os.path.join(os.getcwd(),"artifacts","dataset.zip")
    dataset_path = os.path.join(os.getcwd(),"artifacts")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        try:
            github_url = "https://github.com/deepakpw234/Project-Datasets/raw/refs/heads/main/credit_risk_dataset.zip"

            urllib.request.urlretrieve(github_url,self.data_ingestion_config.zip_dataset_path)

            with zipfile.ZipFile(self.data_ingestion_config.zip_dataset_path,"r") as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.dataset_path)

        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    a = DataIngestion()
    a.initiate_data_ingestion()
