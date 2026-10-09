
import os

import joblib
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)
model = joblib.load("model.pkl")

HTML = """
<h1>Celsius to Fahrenheit Predictor</h1>
<form method="POST">
  <input type="number" name="celsius" step="any"
         placeholder="Enter Celsius" required>
  <button type="submit">Predict</button>
</form>
{% if result is not none %}
  <h2>Predicted Fahrenheit: {{ result }} °F</h2>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        try:
            celsius = float(request.form["celsius"])
            prediction = model.predict(np.array([[celsius]]))
            result = round(float(prediction[0]), 2)
        except (ValueError, KeyError):
            return "Invalid temperature input", 400
    return render_template_string(HTML, result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
