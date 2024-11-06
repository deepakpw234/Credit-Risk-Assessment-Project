import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

class DataCleaningConfig:
    raw_dataset_path = os.path.join(os.getcwd(),"artifacts","credit_risk_dataset.csv")
    clean_dataset_path = os.path.join(os.getcwd(),"artifacts","cleaned_dataset.csv")


class DataCleaning:
    def __init__(self):
        self.data_cleaning_config = DataCleaning()

    