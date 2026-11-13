from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("sleep_model.pkl")
encoder = joblib.load("encoders.pkl")

print("Model and encoder loaded successfully!")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = [
        data["feature1"],
        data["feature2"],
        data["feature3"],
        data["feature4"]
    ]

    prediction = model.predict([features])[0]
    result = encoder.inverse_transform([prediction])[0]

    return jsonify({
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)