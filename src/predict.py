import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR,"models", "best_model.pkl")
scaler_path = os.path.join(BASE_DIR,"models", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def predict(data):
    data = scaler.transform(data)
    return model.predict(data)