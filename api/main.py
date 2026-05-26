# =========================================================
# SAHABATSTOK - FASTAPI DEPLOYMENT
# FILE: main.py
# =========================================================

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# =========================================================
# LOAD MODEL
# =========================================================

#model = joblib.load("best_model_weekly_sales-appliances.pkl")

# =========================================================
# INIT FASTAPI
# =========================================================

app = FastAPI(
    title="SahabatStok API",
    description="API Prediksi Penjualan & Rekomendasi Stock",
    version="1.0"
)

# =========================================================
# INPUT SCHEMA
# =========================================================

class ForecastRequest(BaseModel):

    lag_1: float
    lag_2: float
    lag_3: float

    rolling_mean_3: float
    rolling_mean_6: float

    rolling_std_3: float
    rolling_std_6: float

# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def home():

    return {
        "message": "SahabatStok API Running"
    }

# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict(data: ForecastRequest):

    # =====================================================
    # CONVERT TO DATAFRAME
    # =====================================================

    input_data = pd.DataFrame([{

        "lag_1": data.lag_1,
        "lag_2": data.lag_2,
        "lag_3": data.lag_3,

        "rolling_mean_3": data.rolling_mean_3,
        "rolling_mean_6": data.rolling_mean_6,

        "rolling_std_3": data.rolling_std_3,
        "rolling_std_6": data.rolling_std_6

    }])

    # =====================================================
    # PREDICT
    # =====================================================

    prediction = model.predict(input_data)

    predicted_qty = round(float(prediction[0]), 2)

    # =====================================================
    # SAFETY STOCK
    # =====================================================

    safety_stock = round(
        predicted_qty * 0.2,
        2
    )

    # =====================================================
    # RECOMMENDED STOCK
    # =====================================================

    recommended_stock = round(
        predicted_qty + safety_stock,
        2
    )

    # =====================================================
    # RETURN RESPONSE
    # =====================================================

    return {

        "predicted_order_quantity": predicted_qty,

        "safety_stock": safety_stock,

        "recommended_stock": recommended_stock

    }

# =========================================================
# MULTI-WEEK FORECAST
# =========================================================

@app.post("/forecast/{weeks}")
def multi_forecast(
    weeks: int,
    data: ForecastRequest
):

    forecasts = []

    current_lag_1 = data.lag_1
    current_lag_2 = data.lag_2
    current_lag_3 = data.lag_3

    for week in range(1, weeks + 1):

        input_data = pd.DataFrame([{

            "lag_1": current_lag_1,
            "lag_2": current_lag_2,
            "lag_3": current_lag_3,

            "rolling_mean_3": data.rolling_mean_3,
            "rolling_mean_6": data.rolling_mean_6,

            "rolling_std_3": data.rolling_std_3,
            "rolling_std_6": data.rolling_std_6

        }])

        prediction = model.predict(input_data)

        predicted_qty = round(
            float(prediction[0]),
            2
        )

        safety_stock = round(
            predicted_qty * 0.2,
            2
        )

        recommended_stock = round(
            predicted_qty + safety_stock,
            2
        )

        forecasts.append({

            "week": week,

            "predicted_order_quantity":
                predicted_qty,

            "safety_stock":
                safety_stock,

            "recommended_stock":
                recommended_stock

        })

        # =============================================
        # UPDATE LAG
        # =============================================

        current_lag_3 = current_lag_2
        current_lag_2 = current_lag_1
        current_lag_1 = predicted_qty

    return {
        "forecast_horizon": weeks,
        "results": forecasts
    }
