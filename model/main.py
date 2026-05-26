from fastapi import FastAPI
from pydantic import BaseModel

from pycaret.time_series import load_model
from pycaret.time_series import predict_model

import pandas as pd

app = FastAPI(title="Sales Forecast API-Storage & Organization", version="1.0")

# load model
model = load_model("../model/best_model_produk_p001")


class ForecastRequest(BaseModel):
    n_weeks: int


@app.get("/")
def home():
    return {
        "message": "Forecast API Running"
    }


@app.post("/predict")
def predict_sales(request: ForecastRequest):

    predictions = predict_model(
        model,
        fh=request.n_weeks
    )

    pred_values = predictions["y_pred"].values

    future_dates = pd.date_range(
        start=pd.Timestamp.today(),
        periods=request.n_weeks,
        freq="W"
    )

    result = []

    for date, pred in zip(future_dates, pred_values):

        result.append({
            "week": str(date.date()),
            "forecast": float(f"{pred:.2f}")
        })

    return result