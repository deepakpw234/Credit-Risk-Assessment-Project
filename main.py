import os
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaning
from src.components.data_transformation_01 import DataTransformation1
from src.components.data_transformation_02_undersample import DataTransformationUndersample
from src.components.data_transformation_03_oversample import DataTransformationOversample
from src.components.model_training_01_undersample import ModelTrainingUndersample
from src.components.model_training_02_oversample import ModelTrainingOversample

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
        outlier_removed_undersample_df = data_transformation_undersample.undersample_outlier_removal(undersample_df)

        data_transformation_oversample = DataTransformationOversample()
        oversample_df = data_transformation_oversample.create_oversample_dataframe(df_train_final)
        outlier_removed_oversample_df = data_transformation_oversample.oversample_outlier_removal(oversample_df)


        model_training_undersample = ModelTrainingUndersample()
        model_training_undersample.model_training_undersample(outlier_removed_undersample_df,df_test_final)


        model_training_oversample = ModelTrainingOversample()
        model_training_oversample.model_training_oversample(outlier_removed_oversample_df,df_test_final)

    except Exception as e:
        raise CustomException(e,sys)

