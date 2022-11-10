# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 18:42:12 2022

@author: USUARIO
"""

import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'LGRmodel.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('transport_predictor')

@app.route('/classify', methods=['POST'])
def classify():
    data=request.get_json()
    X=dv.transform([data])
    prediction = model.predict(X)
    result = {
       'Result': bool(prediction[0])
       }
    return jsonify(result)

if __name__ =='__main__':
    app.run(debug=True, host ='0.0.0.0', port=9696)
