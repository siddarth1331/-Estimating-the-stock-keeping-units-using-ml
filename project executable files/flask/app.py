import numpy as np
from flask import Flask, render_template, request
import pickle

# Load the model
model = pickle.load(open("stock.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/y_predict", methods=['POST', 'GET'])
def y_predict():
    # Get input values from the form and convert to float
    x_test = [[float(i) for i in request.form.values()]]
    print(x_test)

    # Predict using the model
    prediction = model.predict(x_test)
    print(prediction)
    pred=prediction[[0]]

    # Render the result template with the prediction
    return render_template('result.html', prediction_text=pred[0])

if __name__ == "__main__":
    app.run()