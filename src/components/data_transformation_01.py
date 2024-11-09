import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

from src.utils import save_object


@dataclass
class DataTransformation1Config:
    preprocessor_path = os.path.join(os.getcwd(),"artifacts","model","preprocessor.pkl")


class DataTransformation1:
    def __init__(self):
        self.data_transformation1_config = DataTransformation1Config()


    def get_data_preprocessor(self):
        try:
            logging.info("Preprocessing started")
            numerical_columns = ['person_age', 'person_income','person_emp_length', 'loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
            categorical_columns = ['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']

            logging.info("columns are divided by their category and saved in varible")

            cat_pipeline = Pipeline(
                steps=[
                ("impute",SimpleImputer(strategy="most_frequent")),
                ("ohe",OneHotEncoder()),
                ("scaling",StandardScaler(with_mean=False))
                ]
            )
            logging.info("catgorical pipeline created")

            num_pipeline = Pipeline(
                steps=[
                ("impute",IterativeImputer(max_iter=20)),
                ("scaling",StandardScaler())
                ]
            )
            logging.info("numerical pipeline created")


            preprocessor = ColumnTransformer([
                ("numerical_pipeline",num_pipeline,numerical_columns),
                ("categorical_pipeline",cat_pipeline,categorical_columns)
            ])
            logging.info("Columns are transformed by columntransformer")


        except Exception as e:
            raise CustomException(e,sys)
        
        return preprocessor
    
    def intitiate_data_transfornmation(self,cleaned_df):
        try:
            logging.info("Data transforamtion Stage 1 is started")
            df_train,df_test = train_test_split(cleaned_df,test_size=0.2,random_state=42)

            logging.info("Train test split performed")
            # Converting in Dataframe
            df_train = pd.DataFrame(df_train,columns=list(cleaned_df.columns))
            df_test = pd.DataFrame(df_test,columns=list(cleaned_df.columns))


            # Creating feature and target dataframes for test and train
            X_train = df_train.drop("loan_status",axis=1)
            y_train = df_train[['loan_status']]

            X_test = df_test.drop("loan_status",axis=1)
            y_test = df_test[['loan_status']]

            logging.info("training and test dataset is created")

            # Getting the preprocessor object  
            preprocessor = self.get_data_preprocessor()

            logging.info("preprocessor is loaded")


            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)


            save_object(
                self.data_transformation1_config.preprocessor_path,
                preprocessor
            )
            logging.info("preprocessor is saved as pickle file")

            logging.info("fit and transform performed")

            # For adding the column name in the array
            # You should always remember one thing = transform column name from columntransformer and column name from pd.get_dummies are always 
            # in the same order , Basically they will remain in the same alphabatically order
            arr_columns = pd.get_dummies(X_train)
            arr_columns.columns

            # Creating dataframe from transformed array
            X_train_scaled = pd.DataFrame(X_train_arr,columns=list(arr_columns.columns))
            X_test_scaled = pd.DataFrame(X_test_arr,columns=list(arr_columns.columns))


            # Restting the index of target feature
            y_train.reset_index(drop=True,inplace=True)
            y_test.reset_index(drop=True,inplace=True)
            

            df_train_final = pd.concat([X_train_scaled,y_train],axis=1)
            

            df_test_final = pd.concat([X_test_scaled,y_test],axis=1)
            
            logging.info("Data transforamtion Stage 1 is completed")



        except Exception as e:
            raise CustomException(e,sys)
        
        return df_train_final,df_test_final