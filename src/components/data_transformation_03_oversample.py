import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass


from src.exception import CustomException
from src.logger import logging

from imblearn.over_sampling import SMOTE


@dataclass
class DataTransformationOversampleConfig:
    oversample_path = os.path.join(os.getcwd(),"artifacts")


class DataTransformationOversample:
    def __init__(self):
        self.data_transformation_oversample_config = DataTransformationOversampleConfig()


    def create_oversample_dataframe(self,df_train_final):
        try:

            oversample_x_train = df_train_final.drop("loan_status",axis=1)
            oversample_y_train = df_train_final[['loan_status']]

            # SMOTE Technique
            sm = SMOTE(sampling_strategy='minority',random_state=42)

            Xsm_train,ysm_train = sm.fit_resample(oversample_x_train,oversample_y_train)

            oversample_df = pd.concat([Xsm_train,ysm_train],axis=1)

            oversample_df = oversample_df.sample(frac=1)
            print(oversample_df)


        except Exception as e:
            raise CustomException(e,sys)
        
        return oversample_df
    
    def oversample_outlier_removal(self,oversample_df):
        try:
            logging.info("Outlier removal process for oversample is started")
            oversample_corr = oversample_df.corr()  # This correlation coeficient relation will give us the interdependency of all features.

            # loan_int_rate and loan_percent_income columns are strongly positively related to loan_status. 
            # So, we will remove outlier from these columns.

            # Outlier removal from loan_int_rate
            defaulter = oversample_df[oversample_df['loan_status']==1]
            print(len(defaulter))
            q25, q75 = np.percentile(defaulter['loan_int_rate'],25), np.percentile(defaulter['loan_int_rate'],75)
            print(f"25th percentile: {q25}, 75th percentile: {q75}")

            iqr = (q75-q25)
            cutoff = 1.5*iqr
            print(f"Inter quartile range is: {iqr}")
            print(f"Cutoff: {cutoff}")

            loan_int_rate_lower = q25-cutoff
            loan_int_rate_upper = q75+cutoff

            print(f"Lower Cutoff: {loan_int_rate_lower}, Upper Cutoff: {loan_int_rate_upper}")

            outliers = [x for x in defaulter['loan_int_rate'] if x<loan_int_rate_lower or x>loan_int_rate_upper]
            print(f"Number of outliers: {len(outliers)}")
            print(f"outliers: {outliers}")

            oversample_df = oversample_df.drop(oversample_df[((oversample_df['loan_int_rate']>loan_int_rate_upper) | (oversample_df['loan_int_rate']<loan_int_rate_lower))].index,axis=0)
            print(oversample_df)

            # Outlier removal from loan_percent_income
            defaulter = oversample_df[oversample_df['loan_status']==1]
            print(len(defaulter))
            q25, q75 = np.percentile(defaulter['loan_percent_income'],25), np.percentile(defaulter['loan_percent_income'],75)
            print(f"25th percentile: {q25}, 75th percentile: {q75}")

            iqr = (q75-q25)
            cutoff = 1.5*iqr
            print(f"Inter quartile range is: {iqr}")
            print(cutoff)

            loan_percent_income_lower = q25-cutoff
            loan_percent_income_upper = q75+cutoff

            print(f"Lower Cutoff: {loan_percent_income_lower}, Upper Cutoff: {loan_percent_income_upper}")

            outliers = [x for x in defaulter['loan_percent_income'] if x<loan_percent_income_lower or x>loan_percent_income_upper]
            print(f"Number of outliers: {len(outliers)}")
            print(f"outliers: {outliers}")

            oversample_df = oversample_df.drop(oversample_df[((oversample_df['loan_percent_income']>loan_percent_income_upper) | (oversample_df['loan_percent_income']<loan_percent_income_lower))].index,axis=0)
            print(oversample_df)


        except Exception as e:
            raise CustomException(e,sys)
        
        return oversample_df
        
        
