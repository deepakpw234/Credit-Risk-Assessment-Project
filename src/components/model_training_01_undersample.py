import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report,roc_auc_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,cross_val_score,cross_val_predict


import warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelTrainingUndersampleConfig:
    model_path = os.path.join(os.getcwd(),"artifacts","model")


class ModelTrainingUndersample:
    def __init__(self):
        self.model_training_undersample_config = ModelTrainingUndersampleConfig()


    def model_training_undersample(self,undersample_df,df_test_final):
        try:
            logging.info("Model training for undersample is started")
            undersample_x_train = undersample_df.drop("loan_status",axis=1)
            undersample_y_train = undersample_df["loan_status"]
            print(undersample_x_train)

            # Converting into array
            undersample_x_train = undersample_x_train.values
            undersample_y_train = undersample_y_train.values


            models = {
                "Logistic Regression":LogisticRegression(),
                "RandomForest Classifier":RandomForestClassifier(),
                "KNeighbors Classifier": KNeighborsClassifier(),
                "SVC":SVC(),
                "DecisionTree Classifier":DecisionTreeClassifier(),
                "CatBoost Classifier":CatBoostClassifier(),
                "XGB Classifier":XGBClassifier(),
                "AdaBoost Classifier":AdaBoostClassifier(),
                "GradientBoosting Classifier":GradientBoostingClassifier()
            }

            logging.info("All classifier model object created")

            # Hyper Parameters Tunning
            logistic_regression_parameters ={
                "penalty": ['l2'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            }
            logistic_regression_grid = GridSearchCV(LogisticRegression(solver="lbfgs",max_iter=1000),param_grid=logistic_regression_parameters)
            logistic_regression_grid.fit(undersample_x_train,undersample_y_train)
            logistic_regression_best_parameters = logistic_regression_grid.best_estimator_
            print(f"Best parameters for Logistic Regression: {logistic_regression_best_parameters}")


            random_forest_parameters = { 
                'n_estimators': [25, 50, 100, 150], 
                'max_features': ['sqrt', 'log2'], 
                'max_depth': [3, 6, 9], 
                'max_leaf_nodes': [3, 6, 9], 
            }
            random_forest_grid = GridSearchCV(RandomForestClassifier(),param_grid=random_forest_parameters)
            random_forest_grid.fit(undersample_x_train,undersample_y_train)
            random_forest_best_parameter = random_forest_grid.best_estimator_
            print(f"Best parameters for RandomForest Classifier: {random_forest_best_parameter}")


            knear_parameters = {
                "n_neighbors": list(range(2,5,1)),
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'] 
            }
            knear_grid = GridSearchCV(KNeighborsClassifier(),param_grid=knear_parameters)
            knear_grid.fit(undersample_x_train,undersample_y_train)
            knear_best_parameters = knear_grid.best_estimator_
            print(f"Best parameters for KNeighbors Classifier: {knear_best_parameters}")


            svm_parameters = {
                'C': [0.5, 0.7, 0.9, 1],
                'kernel': ['rbf', 'poly', 'sigmoid', 'linear']
            }
            svm_grid = GridSearchCV(SVC(),param_grid=svm_parameters)
            svm_grid.fit(undersample_x_train,undersample_y_train)
            svm_best_parameters = svm_grid.best_estimator_
            print(f"Best parameters for SVC: {svm_best_parameters}")


            tree_parameters = {
                "criterion":["gini", "entropy"],
                "splitter":['best','random'],
                'max_depth':[3,4,5,6],
                'min_samples_split':list(range(8, 20, 2)),
                'min_samples_leaf':[5,6,7],
            }
            tree_grid = GridSearchCV(DecisionTreeClassifier(),param_grid=tree_parameters)
            tree_grid.fit(undersample_x_train,undersample_y_train)
            tree_best_parameters = tree_grid.best_estimator_
            print(f"Best parameters for DecisionTree Classifier: {tree_best_parameters}")


            catboost_parameters = {
                "iterations":[50,100,200],
                "learning_rate":[0.01,0.1,0.2,0.5,0.9],
                "depth":[3,6,9],
                "loss_function":["Logloss"]
            }
            catboost_grid = GridSearchCV(CatBoostClassifier(verbose=False),param_grid=catboost_parameters)
            catboost_grid.fit(undersample_x_train,undersample_y_train)
            catboost_best_parameters = catboost_grid.best_estimator_
            print(f"Best parameters for Catboost Classifier: {catboost_best_parameters}")


            XGBClassifier(verbose=False).fit(undersample_x_train,undersample_y_train)
            AdaBoostClassifier().fit(undersample_x_train,undersample_y_train)
            GradientBoostingClassifier().fit(undersample_x_train,undersample_y_train)

            logging.info("Models hyper tunned")

            # Checking for overfitting case by cross_val_score
            print("="*60)
            logistic_regression_cross_val_score = cross_val_score(logistic_regression_best_parameters,undersample_x_train,undersample_y_train,cv=5)
            print(f"logistic_regression_cross_val_score: {logistic_regression_cross_val_score.mean()}")

            RandomForest_Classifier_cross_val_score = cross_val_score(random_forest_best_parameter,undersample_x_train,undersample_y_train,cv=5)
            print(f"RandomForest_Classifier_cross_val_score: {RandomForest_Classifier_cross_val_score.mean()}")

            knear_Classifier_cross_val_score = cross_val_score(knear_best_parameters,undersample_x_train,undersample_y_train,cv=5)
            print(f"knear_Classifier_cross_val_score: {knear_Classifier_cross_val_score.mean()}")

            svm_Classifier_cross_val_score = cross_val_score(svm_best_parameters,undersample_x_train,undersample_y_train,cv=5)
            print(f"svm_Classifier_cross_val_score: {svm_Classifier_cross_val_score.mean()}")

            DecisionTree_Classifier_cross_val_score = cross_val_score(tree_best_parameters,undersample_x_train,undersample_y_train,cv=5)
            print(f"DecisionTree_Classifier_cross_val_score: {DecisionTree_Classifier_cross_val_score.mean()}")

            CatBoostClassifier_cross_val_score = cross_val_score(catboost_best_parameters,undersample_x_train,undersample_y_train,cv=10)
            print(f"CatBoostClassifier_cross_val_score: {CatBoostClassifier_cross_val_score.mean()}")

            XGB_Classifier_cross_val_score = cross_val_score(XGBClassifier(),undersample_x_train,undersample_y_train,cv=10)
            print(f"XGB_Classifier_cross_val_score: {XGB_Classifier_cross_val_score.mean()}")

            AdaBoost_Classifier_cross_val_score = cross_val_score(AdaBoostClassifier(),undersample_x_train,undersample_y_train,cv=5)
            print(f"AdaBoost_Classifier_cross_val_score: {AdaBoost_Classifier_cross_val_score.mean()}")

            GradientBoosting_Classifier_cross_val_score = cross_val_score(GradientBoostingClassifier(),undersample_x_train,undersample_y_train,cv=5)
            print(f"GradientBoosting_Classifier_cross_val_score: {GradientBoosting_Classifier_cross_val_score.mean()}")
            print("="*60)

            logging.info("Cross validation score checked")

            # Checking for best ROC_Auc_score
            logistic_regression_cross_val_predict = cross_val_predict(logistic_regression_best_parameters,undersample_x_train,undersample_y_train,cv=5)
            print(f"logistic_regression_ROC_AUC_score: {roc_auc_score(undersample_y_train,logistic_regression_cross_val_predict)}")

            RandomForest_Classifier_cross_val_predict = cross_val_predict(random_forest_best_parameter,undersample_x_train,undersample_y_train,cv=5)
            print(f"RandomForest_Classifier_ROC_AUC_score: {roc_auc_score(undersample_y_train,RandomForest_Classifier_cross_val_predict)}")

            knear_Classifier_cross_val_predict = cross_val_predict(knear_best_parameters,undersample_x_train,undersample_y_train,cv=5)
            print(f"knear_Classifier_ROC_AUC_score: {roc_auc_score(undersample_y_train,knear_Classifier_cross_val_predict)}")

            svm_Classifier_cross_val_predict = cross_val_predict(svm_best_parameters,undersample_x_train,undersample_y_train,cv=5)
            print(f"svm_Classifier_ROC_AUC_score: {roc_auc_score(undersample_y_train,svm_Classifier_cross_val_predict)}")

            DecisionTree_Classifier_cross_val_predict = cross_val_predict(tree_best_parameters,undersample_x_train,undersample_y_train,cv=5)
            print(f"DecisionTree_Classifier_ROC_AUC_score: {roc_auc_score(undersample_y_train,DecisionTree_Classifier_cross_val_predict)}")

            CatBoostClassifier_cross_val_predict = cross_val_predict(catboost_best_parameters,undersample_x_train,undersample_y_train,cv=10)
            print(f"CatBoostClassifier_ROC_AUC_score: {roc_auc_score(undersample_y_train,CatBoostClassifier_cross_val_predict)}")

            XGB_Classifier_cross_val_predict = cross_val_predict(XGBClassifier(),undersample_x_train,undersample_y_train,cv=10)
            print(f"XGB_Classifier_ROC_AUC_score: {roc_auc_score(undersample_y_train,XGB_Classifier_cross_val_predict)}")

            AdaBoost_Classifier_cross_val_predict = cross_val_predict(AdaBoostClassifier(),undersample_x_train,undersample_y_train,cv=5)
            print(f"AdaBoost_Classifier_ROC_AUC_score: {roc_auc_score(undersample_y_train,AdaBoost_Classifier_cross_val_predict)}")

            GradientBoosting_Classifier_cross_val_predict = cross_val_predict(GradientBoostingClassifier(),undersample_x_train,undersample_y_train,cv=5)
            print(f"GradientBoosting_Classifier_ROC_AUC_score: {roc_auc_score(undersample_y_train,GradientBoosting_Classifier_cross_val_predict)}")
            print("="*60)

            logging.info("ROC_AUC score checked")


            # From the above cross validation and ROC_AUC score,we find that xgboost classifier has very good accuracy.
            # so we will use xgboost classifier for our final prediction

            original_x_test = df_test_final.drop("loan_status",axis=1)
            original_y_test = df_test_final['loan_status']

            # Converting into array
            original_x_test = original_x_test.values
            original_y_test = original_y_test.values

            xgb = XGBClassifier().fit(undersample_x_train,undersample_y_train)

            undersample_prediction = xgb.predict(original_x_test)

            print("Undersample predictions:-")
            print(f"Accuracy score for Undersample: {accuracy_score(original_y_test,undersample_prediction)}")
            print(f"Precision score for Undersample: {precision_score(original_y_test,undersample_prediction)}")
            print(f"Recall score for Undersample: {recall_score(original_y_test,undersample_prediction)}")
            print(f"f1 score for Undersample: {f1_score(original_y_test,undersample_prediction)}")
            print(f"Confusion matrix for Undersample:\n{confusion_matrix(original_y_test,undersample_prediction)}")
            print(f"classification report for Undersample:\n {classification_report(original_y_test,undersample_prediction)}")
            print("="*60)

            logging.info("Undersample model predicted")

            logging.info("Model training for undersample is completed")


        except Exception as e:
            raise CustomException(e,sys)