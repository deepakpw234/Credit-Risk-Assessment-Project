import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

import warnings
warnings.filterwarnings("ignore")

@dataclass
class ModelTrainingOversampleConfig:
    model_path = os.path.join(os.getcwd(),"artifacts","model","model.pkl")


class ModelTrainingOversample:
    def __init__(self):
        self.model_training_oversample_config = ModelTrainingOversampleConfig()


    def model_training_oversample(self,oversample_df,df_test_final):
        try:
            logging.info("Model training for oversample is started")

            oversample_xsm_train = oversample_df.drop("loan_status",axis=1)

            oversample_ysm_train = oversample_df['loan_status']
            


            # converting into Array
            oversample_xsm_train = oversample_xsm_train.values
            oversample_ysm_train = oversample_ysm_train.values


            original_x_test = df_test_final.drop("loan_status",axis=1)
            original_y_test = df_test_final['loan_status']

            # Converting into array
            original_x_test = original_x_test.values
            original_y_test = original_y_test.values


            xgb_oversample = XGBClassifier()
            xgb_oversample.fit(oversample_xsm_train,oversample_ysm_train)

            logging.info("Trained the model")

            save_object(
                self.model_training_oversample_config.model_path,
                xgb_oversample
            )

            logging.info("Oversample model saved")


            oversample_prediction = xgb_oversample.predict(original_x_test)
            print(f"Accuracy score for Oversample: {accuracy_score(original_y_test,oversample_prediction)}")
            print(f"Precision score for Oversample: {precision_score(original_y_test,oversample_prediction)}")
            print(f"Recall score for Oversample: {recall_score(original_y_test,oversample_prediction)}")
            print(f"f1 score for Oversample: {f1_score(original_y_test,oversample_prediction)}")
            print(f"Confusion matrix for Oversample:\n{confusion_matrix(original_y_test,oversample_prediction)}")
            print(f"classification report for Oversample:\n {classification_report(original_y_test,oversample_prediction)}")

            '''
            ========================================
            Oversample :-
            Accuracy score for Oversample: 0.9335342639593909
            Precision score for Oversample: 0.9403578528827038
            Recall score for Oversample: 0.724904214559387
            f1 score for Oversample: 0.818693206404154
            Oversample Confusion Matrix
            [[4955   44]
            [ 364  941]]
            ========================================
            '''

            logging.info('Model training for oversample is completed"')

        except Exception as e:
            raise CustomException(e,sys)