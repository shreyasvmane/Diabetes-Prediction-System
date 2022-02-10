import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
 
app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index2.html')

@app.route('/predict', methods =["GET", "POST"])
def predict():
    if request.method == "POST":
        Name = request.form.get('Name')
        Gender = request.form.get('Gender')
        Age= request.form.get('Age')
        Glucose= request.form.get('Glucose')
        BloodPressure= request.form.get('BloodPressure')
        Insulin = request.form.get('Insulin')
        BMI = request.form.get('BMI')

        data = [[ Age, Glucose, BloodPressure,  Insulin, BMI]] 
        model = pickle.load(open("classifier.pkl", "rb"))
        pred = model.predict(data)

    return render_template('predict.html', prediction=pred,Name=Name,Gender=Gender,Age=Age,
        Glucose=Glucose,BloodPressure=BloodPressure,Insulin=Insulin,BMI=BMI)

if __name__ == "__main__":
    app.run(debug=True)