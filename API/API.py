from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import cv2
import base64
import os

from utils import load_model_scaler, preprocess_input

app = Flask(__name__)
CORS(app)

data_model, scaler = load_model_scaler()

classification_model = tf.keras.models.load_model("image_models/classification_model_final.keras")
unet_model = tf.keras.models.load_model("image_models/unet_best_model.keras")

class_labels = ["benign", "malignant", "normal"]


@app.route('/predict/data', methods=['POST'])
def predict_data():
    try:
        data = request.get_json()

        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features'}), 400

        features = data['features']

        if len(features) != 30:
            return jsonify({'error': 'Expected 30 features'}), 400

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
        return jsonify({"error": str(e)}), 500


def preprocess_image(img, target_size=(128, 128)):
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_image(img):
    img_input = preprocess_image(img)

    cls_pred = classification_model.predict(img_input)
    cls_index = np.argmax(cls_pred)
    cls_label = class_labels[cls_index]

    mask_pred = unet_model.predict(img_input)[0]
    mask_pred = (mask_pred > 0.5).astype(np.uint8)

    return cls_label, mask_pred


def mask_to_base64(mask):
    mask_img = (mask.squeeze() * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', mask_img)
    return base64.b64encode(buffer).decode('utf-8')


@app.route('/predict/image', methods=['POST'])
def predict_image_api():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        label, mask = predict_image(img)
        mask_base64 = mask_to_base64(mask)

        return jsonify({
            "status": "success",
            "prediction": label,
            "mask": mask_base64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "data_model": "loaded",
        "image_models": "loaded"
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)