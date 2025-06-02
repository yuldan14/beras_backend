from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import requests
import json
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

# CORS Middleware (izinkan semua asal — ganti dengan domain produksi jika perlu)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======= Load ARIMA models (.joblib) =======
model_medium_silinda_arima = joblib.load("medium_silinda_arima.joblib")
model_medium_bapanas_arima = joblib.load("medium_bapanas_arima.joblib")
model_premium_silinda_arima = joblib.load("premium_silinda_arima.joblib")
model_premium_bapanas_arima = joblib.load("premium_bapanas_arima.joblib")

# ======= Load LSTM models (.h5) =======
model_medium_silinda_lstm = load_model("medium_silinda_lstm.h5")
model_medium_bapanas_lstm = load_model("medium_bapanas_lstm.h5")
model_premium_silinda_lstm = load_model("premium_silinda_lstm.h5")
model_premium_bapanas_lstm = load_model("premium_bapanas_lstm.h5")

# ======= Data Source from GitHub =======
def load_historical_data():
    url = "https://raw.githubusercontent.com/yuldan14/beras-dashboard/main/app/data_harga.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Gagal mengambil data JSON dari GitHub")

def get_last_n_data(jenis: str, n: int = 30):
    data = load_historical_data()
    values = [d[jenis] for d in data if jenis in d]
    return values[-n:]

# ======= Prediksi ARIMA =======
def predict_arima(model, steps: int):
    return model.forecast(steps).tolist()

# ======= Prediksi LSTM =======
def predict_lstm(model, jenis: str, steps: int = 1, window_size: int = 30):
    data = get_last_n_data(jenis, window_size)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))

    X_input = np.array(data_scaled).reshape(1, window_size, 1)
    predictions = []

    for _ in range(steps):
        pred = model.predict(X_input, verbose=0)[0][0]
        predictions.append(pred)
        data_scaled = np.append(data_scaled, [[pred]], axis=0)
        X_input = np.array(data_scaled[-window_size:]).reshape(1, window_size, 1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten().tolist()
    return predictions

# ======= Endpoint ARIMA =======
@app.post("/predict/{model_name}")
async def predict_arima_route(model_name: str, request: Request):
    body = await request.json()
    steps_ahead = body.get("steps_ahead", 1)

    model_map = {
        "medium_silinda": model_medium_silinda_arima,
        "medium_bapanas": model_medium_bapanas_arima,
        "premium_silinda": model_premium_silinda_arima,
        "premium_bapanas": model_premium_bapanas_arima,
    }

    model = model_map.get(model_name)
    if model is None:
        return {"error": "Model ARIMA tidak ditemukan"}

    result = predict_arima(model, steps_ahead)
    return {"prediction": result}

# ======= Endpoint LSTM =======
@app.post("/predict_lstm/{model_name}")
async def predict_lstm_route(model_name: str, request: Request):
    body = await request.json()
    steps_ahead = body.get("steps_ahead", 1)

    model_map = {
        "medium_silinda": model_medium_silinda_lstm,
        "medium_bapanas": model_medium_bapanas_lstm,
        "premium_silinda": model_premium_silinda_lstm,
        "premium_bapanas": model_premium_bapanas_lstm,
    }

    model = model_map.get(model_name)
    if model is None:
        return {"error": "Model LSTM tidak ditemukan"}

    predictions = predict_lstm(model, model_name, steps=steps_ahead)
    return {"prediction": predictions}
