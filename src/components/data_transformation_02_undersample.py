import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationUndersampleConfig:
    undersample_path = os.path.join(os.getcwd(),"artifacts")


class DataTransformationUndersample:
    def __init__(self):
        self.data_transformation_undersample = DataTransformationUndersampleConfig()


    def create_undersample_dataframe(self,df_train_final):
        try:
            logging.info("Data transformation for undersample is started")
            df_train_final = df_train_final.sample(frac=1)

            defaulter_df = df_train_final[df_train_final['loan_status']==1]
            non_defaulter_df = df_train_final[df_train_final['loan_status']==0][0:5502]

            undersample_df = pd.concat([defaulter_df,non_defaulter_df],axis=0)

            undersample_df = undersample_df.sample(frac=1)
            
            logging.info("undersample dataframe is created")

        except Exception as e:
            raise CustomException(e,sys)
        
        return undersample_df
    

    def undersample_outlier_removal(self,undersample_df):
        try:
            logging.info("Outlier removal process for undersample is started")
            undersample_corr = undersample_df.corr()  # This correlation coeficient relation will give us the interdependency of featues

            # loan_int_rate and loan_percent_income columns are strongly positively related to loan_status. 
            # So, we will remove outlier from these columns.

            # Outlier removal from loan_int_rate
            defaulter = undersample_df[undersample_df['loan_status']==1]
            
            q25, q75 = np.percentile(defaulter['loan_int_rate'],25), np.percentile(defaulter['loan_int_rate'],75)

            iqr = (q75-q25)
            cutoff = 1.5*iqr

            loan_int_rate_lower = q25-cutoff
            loan_int_rate_upper = q75+cutoff


            outliers = [x for x in defaulter['loan_int_rate'] if x<loan_int_rate_lower or x>loan_int_rate_upper]

            undersample_df = undersample_df.drop(undersample_df[((undersample_df['loan_int_rate']>loan_int_rate_upper) | (undersample_df['loan_int_rate']<loan_int_rate_lower))].index,axis=0)

            logging.info("Outlier removed from loan_int_rate column of undersample")

            # Outlier removal from loan_percent_income
            defaulter = undersample_df[undersample_df['loan_status']==1]
            
            q25, q75 = np.percentile(defaulter['loan_percent_income'],25), np.percentile(defaulter['loan_percent_income'],75)

            iqr = (q75-q25)
            cutoff = 1.5*iqr

            loan_percent_income_lower = q25-cutoff
            loan_percent_income_upper = q75+cutoff


            outliers = [x for x in defaulter['loan_percent_income'] if x<loan_percent_income_lower or x>loan_percent_income_upper]

            undersample_df = undersample_df.drop(undersample_df[((undersample_df['loan_percent_income']>loan_percent_income_upper) | (undersample_df['loan_percent_income']<loan_percent_income_lower))].index,axis=0)
            
            logging.info("Outlier removed from loan_percent_income column of undersample")

            logging.info("Outlier removal process for undersample is completed")

        except Exception as e:
            raise CustomException(e,sys)
        
        return undersample_df