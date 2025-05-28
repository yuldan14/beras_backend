from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model ARIMA saat server start
model_medium_silinda = joblib.load("medium_silinda_arima.joblib")
model_medium_bapanas = joblib.load("medium_bapanas_arima.joblib")
model_premium_silinda = joblib.load("premium_silinda_arima.joblib")
model_premium_bapanas = joblib.load("premium_bapanas_arima.joblib")

class PredictRequest(BaseModel):
    steps_ahead: int

@app.post("/predict/{model_name}")
def predict(model_name: str, req: PredictRequest):
    models = {
        "medium_silinda": model_medium_silinda,
        "medium_bapanas": model_medium_bapanas,
        "premium_silinda": model_premium_silinda,
        "premium_bapanas": model_premium_bapanas,
    }
    if model_name not in models:
        return {"error": "Model not found"}
    pred = models[model_name].predict(n_periods=req.steps_ahead)
    return {"prediction": pred.tolist()}
