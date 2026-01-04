import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

MODEL_PATH = Path(r"../Models\Simple_Models\RandomForestClassifier.joblib")

app = FastAPI(
    title=" Best Dataset Prediction API",
    version="1.0"
)

pipeline = None  


# --------------------------------------------------
# Load model ON STARTUP (NOT import time)
# --------------------------------------------------
@app.on_event("startup")
def load_model():
    global pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Failed to load model:", e)
        pipeline = None


# --------------------------------------------------
# Schemas
# --------------------------------------------------
from pydantic import BaseModel
from typing import Optional

class HotelBookingInput(BaseModel):
    hotel: str
    lead_time: int
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: int
    babies: int
    meal: str
    country: str
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    assigned_room_type: str
    booking_changes: int
    deposit_type: str
    agent: Optional[int] = 0
    company: Optional[int] = 0
    days_in_waiting_list: int
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int
    reservation_status: str
    reservation_status_date: str
    city: str


class PredictionOutput(BaseModel):
    is_canceled: int
    cancellation_probability: float



# --------------------------------------------------
# Health
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# Predict
# --------------------------------------------------
import logging

# Logging konfiguratsiyasi (server boshida)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.post("/predict", response_model=PredictionOutput)
def predict(data: HotelBookingInput):
    if pipeline is None:
        raise RuntimeError("Model not loaded")

    # Input → DataFrame
    df = pd.DataFrame([data.model_dump()])

    # 🔍 DEBUG: logging bilan
    proba_all = pipeline.predict_proba(df)[0]
    logger.debug("DEBUG predict_proba: %s", proba_all)

    # Threshold bilan prediction
    threshold = 0.25
    pred = 1 if proba_all[1] >= threshold else 0

    return PredictionOutput(
        is_canceled=pred,
        cancellation_probability=round(float(proba_all[1]), 4)
    )

