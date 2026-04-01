import os
import pickle
import pandas as pd

def load_model_scaler():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler


def preprocess_input(input_list, scaler):
    input_array = pd.DataFrame([input_list], columns=scaler.feature_names_in_)
    input_scaled = scaler.transform(input_array)
    return input_scaled