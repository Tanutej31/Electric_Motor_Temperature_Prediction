from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(x) for x in request.form.values()]
        scaled_input = scaler.transform([input_data])
        prediction = model.predict(scaled_input)
        output = round(prediction[0], 2)

        return render_template("index.html",
                               prediction_text=f"Predicted Temperature: {output}")
    except:
        return render_template("index.html",
                               prediction_text="Invalid Input")

if __name__ == "__main__":
    app.run(debug=True)