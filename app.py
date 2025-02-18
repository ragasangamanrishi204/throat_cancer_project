from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import joblib  # For loading scaler (optional)

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Load scaler (if used)
scaler = joblib.load("scaler.pkl")  # Ensure you save the scaler during training

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            data = np.array([float(x) for x in request.form.values()]).reshape(1, -1)
            data_scaled = scaler.transform(data)  # Scale input (if used)
            prediction = model.predict(data_scaled)[0][0]
            result = "Cancer Detected" if prediction > 0.5 else "No Cancer"
        except Exception as e:
            result = f"Error: {str(e)}"
        
        return render_template("index.html", prediction=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
