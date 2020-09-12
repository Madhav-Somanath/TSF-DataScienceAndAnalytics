from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model = pickle.load(open('SupervisedML.pkl', 'rb'))
    hours = float(request.form['hours'])
    marks = model.predict([[hours]])
    
    return render_template('index.html', prediction='Your predicted marks are {:.2f}'.format(marks[0]))

if __name__ == "__main__":
    app.run(debug = True)