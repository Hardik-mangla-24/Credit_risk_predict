from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model and threshold
model = joblib.load("credit_risk_model.pkl")
threshold = joblib.load("best_threshold.pkl")

# Create FastAPI app
app = FastAPI()


# Input data
class LoanApplication(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


# Prediction API
@app.post("/predict")
def predict(data: LoanApplication):

    # Input ko DataFrame mein convert
    input_data = pd.DataFrame([data.model_dump()])

    # Probability of class 1
    probability = model.predict_proba(input_data)[0, 1]

    # Prediction
    prediction = int(probability >= threshold)

    return {
        "default_probability": float(probability),
        "default_prediction": prediction,
        "threshold": float(threshold),
        "Result": "High Risk" if prediction == 1 else "Low Risk"
    }