import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass


from src.exception import CustomException
from src.logger import logging

class DataCleaningConfig:
    raw_dataset_path = os.path.join(os.getcwd(),"artifacts","credit_risk_dataset.csv")
    clean_dataset_path = os.path.join(os.getcwd(),"artifacts","cleaned_dataset.csv")


class DataCleaning:
    def __init__(self):
        self.data_cleaning_config = DataCleaningConfig()

    
    def initiate_data_cleaning(self):
        try:
            logging.info("Data cleaning is started")
            credit_risk = pd.read_csv(self.data_cleaning_config.raw_dataset_path)
            logging.info("Dataset is loaded in the dataframe")

            # Dropping the duplicates rows from the dataset
            credit_risk = credit_risk.drop_duplicates()
            logging.info("Duplicate rows are dropped from the dataframe")

            # Deleting unrealistic values from person age and person employment length column
            # Dropping the loan_percent_income column from the dataset as it is simply dividation of loan_percent and income column
            credit_risk = credit_risk[(credit_risk['person_age']>=18) & (credit_risk['person_age']<=80)]
            credit_risk = credit_risk[(credit_risk['person_emp_length']>=0) & (credit_risk['person_emp_length']<=50)] 
            logging.info("Unrealistic values are deleted from the dataframe")


            cleaned_df = credit_risk.drop('loan_percent_income',axis=1)
            '''
            In this cleaned dataset, only null values are left to fill by approx values
            These values are filled with the iterative imputer in data transformation stage
            '''
            logging.info("Data cleaning ended")
        except Exception as e:
            raise CustomException(e,sys)
        
        return cleaned_df



    