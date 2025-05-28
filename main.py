from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ganti dengan domain Next.js kalau sudah production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
model_medium_silinda = joblib.load("medium_silinda_arima.joblib")
model_medium_bapanas = joblib.load("medium_bapanas_arima.joblib")
model_premium_silinda = joblib.load("premium_silinda_arima.joblib")
model_premium_bapanas = joblib.load("premium_bapanas_arima.joblib")

def predict_future(model, steps):
    return model.forecast(steps).tolist()

@app.post("/predict/{model_name}")
async def predict(model_name: str, request: Request):
    body = await request.json()
    steps_ahead = body.get("steps_ahead", 1)

    if model_name == "medium_silinda":
        result = predict_future(model_medium_silinda, steps_ahead)
    elif model_name == "medium_bapanas":
        result = predict_future(model_medium_bapanas, steps_ahead)
    elif model_name == "premium_silinda":
        result = predict_future(model_premium_silinda, steps_ahead)
    elif model_name == "premium_bapanas":
        result = predict_future(model_premium_bapanas, steps_ahead)
    else:
        return {"error": "Model tidak ditemukan"}

    return {"prediction": result}
