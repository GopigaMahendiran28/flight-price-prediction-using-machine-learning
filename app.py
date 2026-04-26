from flask import Flask, request, render_template
import joblib
import numpy as np
import os
import gdown

app = Flask(__name__)

# download model from drive
url = "https://drive.google.com/uc?id=1gU1F_0HhQmk-zBhzoNqKbKvYuqM4_9NT"
output = "model.pkl"

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

model = joblib.load(output)

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
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
