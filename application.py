import os
import sys

from flask import Flask,render_template,url_for,request,redirect

from src.exception import CustomException
from src.logger import logging

application = Flask(__name__ ,template_folder="templates")

app = application


@app.route("/",methods=["GET","POST"])
def index():
    return render_template("index.html")

@app.route("/home",methods = ["GET","POST"])
def credit_risk_prediction():
    if request.method == "GET":
        return render_template("home.html")
    



if __name__=="__main__":
    app.run(debug=True)


