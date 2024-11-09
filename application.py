import os
import sys
import pandas as pd
import numpy as np

from flask import Flask,render_template,url_for,request,redirect

from src.exception import CustomException
from src.logger import logging

from src.pipelines.train_pipeline import TrainingPipeline
from src.pipelines.predict_pipeline import PredictionPipeline

application = Flask(__name__ ,template_folder="templates")

app = application


@app.route("/",methods=["GET","POST"])
def index():
    return render_template("index.html")

@app.route("/home",methods = ["GET","POST"])
def credit_risk_prediction():
    try:
        if request.method == "GET":
            return render_template("home.html")
        
        else:
            data = {
                "person_age": float(request.form.get("age")),
                "person_income": float(request.form.get('income')),
                "person_home_ownership":request.form.get('ownership'),
                "person_emp_length":float(request.form.get('employment_year')),
                "loan_intent":request.form.get('loan_intension'),
                "loan_grade":request.form.get('loan_grade'),
                "loan_amnt":float(request.form.get('loan_amount')),
                "loan_int_rate":float(request.form.get('loan_interest_rate')),
                "cb_person_default_on_file":request.form.get('default_history'),
                "cb_person_cred_hist_length":float(request.form.get('credit_history')),
            }

            user_df = pd.DataFrame(data,index=[0])

            training_pipeline = TrainingPipeline()
            user_df_arr = training_pipeline.user_data_preprocessing(user_df)

            prediction_pipeline = PredictionPipeline()
            prediction = prediction_pipeline.model_prediction(user_df_arr)

            
            return render_template("home.html",result=prediction)
    
    except Exception as e:
        raise CustomException(e,sys)




if __name__=="__main__":
    app.run(debug=True)


