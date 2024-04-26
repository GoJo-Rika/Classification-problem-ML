import json
import pickle

from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
## Load the model
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():

    user_age = [int(x) for x in request.form.values()]
    user_salary = [user_age]
    scaled_result = scaler.tansform(user_salary)
    prediction = model.predict(scaled_result)
    if prediction==1:
        return render_template('home.html', prediction_text='Yes, Purchaser')
    else:
        return render_template('home.html', prediction_text='No, Not a Purchaser')
    
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()
    age = data['age']
    salary = data['salary']
    gender = data['gender']
    user_age = [[age, salary, gender]]
    scaled_result = scaler.transform(user_age)
    res = model.predict(scaled_result)
    if res==1:
        return jsonify({'Sales Prediction': 'Yes, Purchaser'})
    else:
        return jsonify({'Sales Prediction': 'No, Not a Purchaser'})
    
    if __name__ == '__main__':
        app.run(debug=True)