from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaning
from src.components.data_transformation_01 import DataTransformation1


if __name__=="__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()


    data_cleaning = DataCleaning()
    cleaned_dataset = data_cleaning.initiate_data_cleaning()


    data_transformation1 = DataTransformation1()
    data_transformation1.intitiate_data_transfornmation(cleaned_dataset)

