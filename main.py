from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load semua model ARIMA
model_medium_silinda = joblib.load("medium_silinda_arima.joblib")
model_medium_bapanas = joblib.load("medium_bapanas_arima.joblib")
model_premium_silinda = joblib.load("premium_silinda_arima.joblib")
model_premium_bapanas = joblib.load("premium_bapanas_arima.joblib")

class PredictRequest(BaseModel):
    steps_ahead: int  # Prediksi berapa langkah ke depan, misal 1 untuk besok

@app.post("/predict/medium_silinda")
def predict_medium_silinda(req: PredictRequest):
    pred = model_medium_silinda.predict(n_periods=req.steps_ahead)
    return {"prediction": pred.tolist()}

@app.post("/predict/medium_bapanas")
def predict_medium_bapanas(req: PredictRequest):
    pred = model_medium_bapanas.predict(n_periods=req.steps_ahead)
    return {"prediction": pred.tolist()}

@app.post("/predict/premium_silinda")
def predict_premium_silinda(req: PredictRequest):
    pred = model_premium_silinda.predict(n_periods=req.steps_ahead)
    return {"prediction": pred.tolist()}

@app.post("/predict/premium_bapanas")
def predict_premium_bapanas(req: PredictRequest):
    pred = model_premium_bapanas.predict(n_periods=req.steps_ahead)
    return {"prediction": pred.tolist()}
