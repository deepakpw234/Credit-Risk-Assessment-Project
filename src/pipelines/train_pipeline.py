import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation_01 import DataTransformation1
from src.utils import load_object

@dataclass
class TrainingPipelineConfig:
    preprocessor_path = os.path.join(os.getcwd(),"artifacts","model","preprocessor.pkl")


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def user_data_preprocessing(self,user_df):
        try:
            
            user_df.insert(8,"loan_percent_income",user_df['loan_amnt']/user_df['person_income'])

            print(user_df)
            
            preprocessor = load_object(self.training_pipeline_config.preprocessor_path)

            user_df_arr = preprocessor.transform(user_df)


        except Exception as e:
            raise CustomException(e,sys)
        
        return user_df_arr