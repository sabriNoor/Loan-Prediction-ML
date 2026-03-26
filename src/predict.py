import joblib
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR,"models", "best_model.pkl")
scaler_path = os.path.join(BASE_DIR,"models", "scaler.pkl")
col_path = os.path.join(BASE_DIR,"models", "columns.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
expected_cols = joblib.load(col_path)

def preprocess_input(df):
    df = df.copy()

    # encoding
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})

    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)

    # one-hot encoding
    df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)

    # FIX: align columns with training
    df = df.reindex(columns=expected_cols, fill_value=0)

    return df


def predict(data):
    data = preprocess_input(data)

    # scaling
    data = scaler.transform(data)

    # prediction
    return model.predict(data)