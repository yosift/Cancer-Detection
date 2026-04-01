from utils import load_model_scaler, preprocess_input

model, scaler = load_model_scaler()

def predict_breast_cancer_risk(user_input_data):
    input_scaled = preprocess_input(user_input_data, scaler)
    probability = model.predict_proba(input_scaled)[0][1] * 100

    if probability < 30:
        risk_level = "Low Risk"
    elif 30 <= probability <= 70:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    return risk_level, probability

dummy_input = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
               1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 
               0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]

risk, prob = predict_breast_cancer_risk(dummy_input)
print(f"Probability: {prob:.2f}% -> Result: {risk}")