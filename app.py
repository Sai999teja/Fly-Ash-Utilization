from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
import joblib
import sklearn

app = Flask(__name__)

from sklearn.ensemble import GradientBoostingClassifier

model = pickle.load(open('fly_model.pkl', 'rb'))
model2 = pickle.load(open('fly_model2.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    prediction1 = model.predict(final)
    prediction2 = model2.predict(final)
    prediction1 = prediction1[0][0]
    prediction2 = prediction2[0][0]
    prediction1 = str(prediction1)
    prediction2 = str(prediction2)

    return render_template('index.html', pred='Fly Ash Generation and Utilization are as follows', result1=prediction1,result2=prediction2)


if __name__ == '__main__':
    app.run(debug=True)
