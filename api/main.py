from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from src.predict import predict

app = FastAPI()

class LoanRequest(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str


@app.post("/predict")
def predict_api(payload: LoanRequest):

    # convert to dataframe (1 row)
    data = pd.DataFrame([payload.dict()])

    # predict
    pred = predict(data)

    return {
        "prediction": int(pred[0])
    }