

# 1. Credit Risk Assessment Project

The objective of this project is to predict whether a loan applicant will default in the future or not. For this project, dataset is sourced from Kaggle website which contains total loan applicants of 32,581 in which 7,108 are defaulters and 25,473 are non defaulters. After performing detailed EDA, Undersampling and Oversampling (SMOTE) techniques, model achieve a notable accuracy of 93.52%. The confusion matrix for the model is as follows:
                
                    Predicted Value
                       D     ND
        True    D    [941   364
        Value   ND     44  4955]
- 941 - TRUE POSITIVE (model predicted defaulter, actually defaulter)
- 364 - FALSE NEGATIVE (model predicted non defaulter, actually defaulter)
- 44  - FALSE POSITIVE (model predicted defaulter, actually non defaulter)
- 4955 - TRUE NEGATIVE (model predicted non defaulter, actually non defaulter)

This outcome demonstrates the model’s potential effectiveness in helping financial institutions assess the likelihood of loan defaults, making it a valuable tool in risk management and decision-making processes.




# 2. Table of contents 

1. Dataset
2. Data Ingestion
3. Data Cleaning
4. Data Transforamtion
- Jupyter Notebook
5. Model Training
- Under sample training
- Over sample (SMOTE) training
6. Prediction pipeline
7. Flask application


## 2.1 Dataset

The dataset for this project is sourced from Kaggle, comprising 32,581 loan applications, with 7,108 labeled as defaulters and 25,473 as non defaulters. It contains 12 columns, including Age, Income, House ownership, Employment length, Loan intension, Loan grade,Loan amount, Loan interest rate, Loan status, Loan percent income, Default on file, Credit history length. The dataset is imbalanced, with default applications accounting for only 21.8% of the total.

- You can download the dataset by [clicking here](https://github.com/deepakpw234/Project-Datasets/raw/refs/heads/main/credit_risk_dataset.zip)


## 2.2 Data Ingestion


In this section of the project, the dataset is initially downloaded from a GitHub repository and then unzipped to prepare it for subsequent data transformation and analysis. This crucial step ensured that the raw data was both accessible and in the appropriate format for effective preprocessing. This preparation is essential to ensure that the dataset was ready for exploratory data analysis (EDA), enhancing the accuracy and reliability of insights gained from the model.

- You can check out the data ingestion code by [clicking here](https://github.com/deepakpw234/Credit-Risk-Assessment-Project/blob/main/src/components/data_ingestion.py)


## 2.3 Data Cleaning


In this phase of the project, unrealistic values, duplicates, and null entries are removed from the dataset to enhance data quality. Outliers are also identified, particularly among features that showed a strong correlation with the target variable. This careful data cleaning process was essential to ensure accuracy in the subsequent analysis. With these steps completed, the dataset is now fully prepared for data transformation, setting a solid foundation for building a reliable predictive model.

- You can check out the data cleaning code by [clicking here](https://github.com/deepakpw234/Credit-Risk-Assessment-Project/blob/main/src/components/data_cleaning.py)

## 2.4 Data Transforamtion

Separate pipelines are constructed to handle numerical and categorical columns, enabling tailored transformations for each type. A column transformer is then applied to integrate these pipelines and transform the dataset effectively. For handling missing values, an iterative imputer is used to estimate and replace null entries, ensuring consistency across the dataset. Numerical features are standardized with a standard scaler, while categorical columns are converted into numerical values using one-hot encoding. This approach preserve the original categorical information while making it compatible with the model. Finally, the entire processing pipeline is saved as a pickle file, allowing seamless feature transformation during production. This setup ensures that the same preprocessing steps are applied to new data, supporting consistent model predictions in real-world applications.

- You can check out the jupyter notebook by [clicking here](https://github.com/deepakpw234/Credit-Risk-Assessment-Project/blob/main/notebooks/Credit%20Risk%20Analysis(EDA)%20-%20Deepak%20.ipynb)


- You can check out the data transforamtion code by [clicking here](https://github.com/deepakpw234/Credit-Risk-Assessment-Project/blob/main/src/components/data_transformation_01.py)


## 2.5 Model Training

- **Undersampling**-
Undersampling is a technique used to address class imbalance in a dataset. It involves reducing the number of instances from the majority class to make it comparable to the minority class. By doing this, the model gets an equal or more balanced representation of classes.


- **Oversampling (SMOTE)**-
Oversampling is a technique used to address class imbalance in datasets. SMOTE generates synthetic samples for the minority class by creating new instances along the line segments between the minority class instances and their nearest neighbors.

Nine different models are trained and hyper-tuned, XGBoost Classifier performing the best among them. The model’s accuracy was evaluated using the ROC-AUC score and a classification report for the under-sampling and over-sampling techniques.

- You can check out the jupyter notebook by [clicking here](https://github.com/deepakpw234/Credit-Risk-Assessment-Project/blob/main/notebooks/Credit%20Risk%20Analysis(Model%20Training)%20-%20Deepak.ipynb)

- You can check out the model training code by [clicking here](https://github.com/deepakpw234/Credit-Risk-Assessment-Project/blob/main/src/components/model_training_01_undersample.py)



## 2.6 Prediction pipeline

The training and prediction pipeline is designed to efficiently handle new data. In this process, each new entry is first scaled to match the training data format. The saved and trained model is then loaded to make predictions using these scaled values. This approach ensures accurate classification of new application as either defaulter or non defaulter, maintaining consistency and precision in detecting defaults.


## 2.7 Flask Web Framework

The application for users is developed using the Flask web framework. This web-based interface allows users to input necessary application details, which are then processed by the trained model. Based on these inputs, the model predicts whether the applicant is defaulter or non defaulter.
# Project Structure

![credit risk project structure](https://github.com/user-attachments/assets/423ba8d5-515d-4b38-a525-15ed11a795a2)


## Installation

### Prerequisites

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Flask
- keras
- tensorflow
- dill
- imbalanced-learn
- catboost
- xgboost


### Steps

Step 1: Clone the repository
```bash
git clone https://github.com/deepakpw234/Credit-Risk-Assessment-Project.git
```
Step 2: Change into the project directory
```bash
cd Credit-Risk-Assessment-Project
```

Step 3: Create and activate a virtual environment

```bash
conda create -n credit_risk_env python==3.11 -y
```
```bash
conda activate credit_risk_env
```

Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

Step 5: Run the project

```bash
python main.py
```

Step 6: Run the flask application

```bash
python application.py
```
## Live Demo

https://github.com/user-attachments/assets/e93e04c0-2d35-4943-93bd-d80d7c8745fc

# Hi, I'm Deepak Pawar! 👋


## 🚀 About Me
I'm a Data Scientist with over 2 years of experience in leveraging data analytics, statistical modelling, and machine learning to drive actionable insights and business growth. Proficient in leveraging Python, SQL, Scikit-Learn and Machine Learning techniques to solve complex data problems and enhance predictive analytics. Strong background in data preprocessing, feature engineering, and model evaluation, with a proven track record of optimizing model performance and scalability. Also, Expertise in developing and deploying end-to-end data science solutions within CI/CD pipelines, ensuring seamless integration and continuous delivery of models and applications.


## 🛠 Skills

- Languages – Python, C Programming
- Libraries – Pandas, NumPy, Scikit-Learn, TensorFlow, Keras, Transformers, Hugging face Library, Neural Netowrk
- Visualization Tools – Matplotlib, Seaborn, Power BI
- Databases – SQL, MongoDB
- Clouds – Amazon Web Service (AWS), Microsoft Azure
- Misc – GitHub Action, Docker, Flask, Jupyter Notebook, Office 365


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/deepakpw234)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/deepak-pawar-92a2a5b5/)



## Author

- [@deepakpw234](https://github.com/deepakpw234)






