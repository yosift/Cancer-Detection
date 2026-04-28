from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from utils import load_model_scaler, preprocess_input

app = Flask(__name__)
CORS(app)

data_model, scaler = load_model_scaler()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "Cancer Detection API is running"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "data_model": "loaded"
    })


@app.route("/predict/data", methods=["POST"])
def predict_data():
    try:
        data = request.get_json()

        if not data or "features" not in data:
            return jsonify({"error": "Missing features"}), 400

        features = data["features"]

        if len(features) != 30:
            return jsonify({"error": "Expected 30 features"}), 400

        input_scaled = preprocess_input(features, scaler)

        probability = data_model.predict_proba(input_scaled)[0][1] * 100

        if probability < 30:
            risk_level = "Low Risk"
        elif probability <= 70:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        return jsonify({
            "status": "success",
            "probability": round(probability, 2),
            "risk_level": risk_level,
            "prediction": "Malignant" if probability >= 50 else "Benign"
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
