from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaning


if __name__=="__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()


    data_cleaning = DataCleaning()
    data_cleaning.initiate_data_cleaning()