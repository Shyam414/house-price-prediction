import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from the form (all 13 features)
        features = [
            float(request.form["CRIM"]),
            float(request.form["ZN"]),
            float(request.form["INDUS"]),
            float(request.form["CHAS"]),
            float(request.form["NOX"]),
            float(request.form["RM"]),
            float(request.form["AGE"]),
            float(request.form["DIS"]),
            float(request.form["RAD"]),
            float(request.form["TAX"]),
            float(request.form["PTRATIO"]),
            float(request.form["B"]),
            float(request.form["LSTAT"])
        ]

        # Convert to NumPy array and scale input
        features_scaled = scaler.transform([features])

        # Make prediction
        predicted_price = model.predict(features_scaled)[0]

        return render_template("index.html", prediction_text=f"Predicted House Price: ${predicted_price:,.2f}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
