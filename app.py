from flask import Flask, request, render_template
import joblib
import numpy as np
import requests

app = Flask(__name__)

# 🔥 Download model from Google Drive
url = "https://drive.google.com/uc?id=1gU1F_0HhQmk-zBhzoNqKbKvYuqM4_9NT"

response = requests.get(url)
open("model.pkl", "wb").write(response.content)

# Load model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    input_features = [
        float(data['airline']),
        float(data['source']),
        float(data['destination']),
        float(data['stops']),
        float(data['duration'])
    ]

    features = np.array([input_features])
    prediction = model.predict(features)

    return render_template("index.html", prediction_text="Predicted Price: {}".format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)