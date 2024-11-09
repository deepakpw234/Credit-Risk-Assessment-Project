import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.utils import load_object

@dataclass
class PredictionPipelineConfig:
    model_path = os.path.join(os.getcwd(),"artifacts","model","model.pkl")


class PredictionPipeline:
    def __init__(self):
        self.prediction_pipeline_config = PredictionPipelineConfig()

    def model_prediction(self,user_df_arr):
        try:
            
            model = load_object(self.prediction_pipeline_config.model_path)

            y_pred = model.predict(user_df_arr)


            if y_pred[0]==1:
                prediction = "Defaulter"
            
            else:
                prediction = "Non defaulter"
            
            print(prediction)

        except Exception as e:
            raise CustomException(e,sys)
        
        return prediction