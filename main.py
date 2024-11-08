import os
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaning
from src.components.data_transformation_01 import DataTransformation1
from src.components.data_transformation_02_undersample import DataTransformationUndersample
from src.components.data_transformation_03_oversample import DataTransformationOversample

from src.exception import CustomException
from src.utils import logging

if __name__=="__main__":
    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()


        data_cleaning = DataCleaning()
        cleaned_dataset = data_cleaning.initiate_data_cleaning()


        data_transformation1 = DataTransformation1()
        df_train_final,df_test_final =data_transformation1.intitiate_data_transfornmation(cleaned_dataset)


        data_transformation_undersample = DataTransformationUndersample()
        undersample_df = data_transformation_undersample.create_undersample_dataframe(df_train_final)
        data_transformation_undersample.undersample_outlier_removal(undersample_df)

        data_transformation_oversample = DataTransformationOversample()
        oversample_df = data_transformation_oversample.create_oversample_dataframe(df_train_final)
        data_transformation_oversample.oversample_outlier_removal(oversample_df)

    except Exception as e:
        raise CustomException(e,sys)

