import utils
import numpy as np
from flask import Flask, render_template, request,url_for

app=Flask(__name__)

@app.route('/')
def home():

	return render_template('app.html')



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']
        
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        for value in data:
            if type(value)==float:
                pass
            else:
                print("error")
        my_prediction = utils.load_model(data)
        print(my_prediction)

       
        
        return render_template('results.html', prediction=my_prediction)

