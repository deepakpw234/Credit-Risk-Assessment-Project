import os
import sys
from pathlib import Path

list_of_file = [
    "src/__init__.py",
    "src/exception.py",
    "src/logger.py",
    "src/utils.py",
    "artifacts/abc.txt",
    "logs/abc.txt",
    "src/compontents/__init__.py",
    "src/compontents/data_ingestion.py",
    "src/compontents/data_cleaning.py",
    "src/compontents/data_transforamtion.py",
    "src/compontents/model_training.py",
    "src/compontents/neural_network.py",
    "src/pipelines/__init__.py",
    "src/pipelines/train_pipeline.py",
    "src/pipelines/predict_pipeline.py",
    "src/templates/index.html",
    "src/templates/home.html",
    "application.py",
    "main.py",
    "Dockerfile",
    ".ebextensions/python.config",
]


for i in list_of_file:
    file_path = Path(i)

    directory ,file_name = os.path.split(file_path)

    if directory != "":
        os.makedirs(directory,exist_ok=True)

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):

        with open(file_path,"w") as file:
            pass
