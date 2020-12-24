# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 23:29:32 2020

@author: Huy
"""


from flask import Flask, render_template, session, redirect, url_for, session, request

import numpy as np  
import joblib
import json


# Get model list
filename = 'stt_model.json'
with open(filename,'r') as modelist:
    stt_model = json.load(modelist)
    
app = Flask(__name__)


def matching(fts_model, fts_input):
    for i in fts_model:
        if str(fts_input) == i:
            return stt_model[i]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    # Get json
    dic = {}
    for key, value in request.form.items():
        if value != "":
            dic[key] = value
    
    fts_input = [i for i in dic.keys()]
    fts_model = [i for i in stt_model.keys()]
    # match features
    stt = matching(fts_model, fts_input)
    path = 'rfr_model_'+str(stt)+'.pkl'
    model = joblib.load(path)

        
    a = dic
    features = [i for i in dic.values()]
    
    # predict
    prediction = model.predict([features])
    output = round(prediction[0])
    return render_template('index.html',
                           features = a,
                           prediction_text='Rented bike count should be {}'.format(output))
if __name__ == "__main__":
    app.run(debug = True)
