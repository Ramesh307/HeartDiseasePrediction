
from flask import Flask, request, render_template
import pandas as pd
from utils.Function import predict_rf

import sklearn

app = Flask(__name__)

heartdiseasemodel = pd.read_pickle("G:\HeartDiseasePrediction\HeartDiseasePrediction/utils/model_tree.pickle")


@app.route('/diagnose')
def diagnose():
    return render_template("index2.html")


@app.route('/')
def home():
    return render_template("Layout.html")


@app.route('/description')
def description():
    return render_template("description.html")


@app.route('/about')
def about():
    return render_template("About.html")


@app.route('/predict_disease', methods=['POST'])
def predict():
    
    if request.method == 'POST':
        age = int(request.form.get("age"))
        gender = int(request.form.get("gender"))
        cp = int(request.form.get("chest"))
        trestbps = int(request.form.get("trestbps"))
        chol = int(request.form.get("chol"))
        fbs = int(request.form.get("fbs"))
        restecg = int(request.form.get("restecg"))
        thalach = int(request.form.get("thalach"))
        exang = int(request.form.get("exang"))
        oldpeak = float(request.form.get("oldpeak"))
        test = {
            'age': age,
            'sex': gender,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak
        }
        test = pd.DataFrame(test, index=[0])
        heart_prediction = predict_rf(heartdiseasemodel, test)
        if (heart_prediction[0] == 1):
            result = "You have Heart Disease"
            color = "red"
        else:
            result = "You don't have Heart Disease"
            color = "green"
        print(result)

        return render_template("index2.html", prediction=result, color=color)
    else:
        return render_template("index2.html")


if __name__ == '__main__':
    app.run()
