
import numpy as np

from flask import Flask, render_template, request
import pickle

from pandas import array


app = Flask(__name__)


model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def start():
    return render_template('start.html')


@app.route("/main")
def main():
    return render_template('mainpage.html')


@app.route("/result")
def result():
    return render_template('after.html')


@app.route("/predict", methods=['POST', 'GET'])
def predict():

    totalfat = request.form['totalfat']
    colestrol = request.form['colestrol']
    sodium = request.form['sodium']
    totalcarbohydrates = request.form['totalcarbohydrates']
    protein = request.form['protein']
    calcium = request.form['calcium']
    potassium = request.form['potassium']
    iron = request.form['iron']

    arr = np.array(
        [[totalfat, colestrol, sodium, totalcarbohydrates, protein,   calcium,  potassium, iron]])
    pred = model.predict(arr)

    return render_template('after.html', f=totalfat, c=colestrol, s=sodium, carbs=totalcarbohydrates, pro=protein, cal=calcium, p=potassium, i=iron, data=str(pred).replace('[', '').replace(']', ''))


if __name__ == "__main__":
    app.run(debug=True, port=8000)
