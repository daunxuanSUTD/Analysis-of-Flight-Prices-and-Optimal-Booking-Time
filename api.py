from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model.joblib"

app = FastAPI(
    title="Flight Price Prediction API",
    description="Predict flight ticket prices based on a trained log-price model.",
    version="1.0.0",
)

# Load the full pipeline (preprocessing + model)
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    airline: str
    source_city: str
    destination_city: str
    departure_time: str          # e.g. "Morning", "Evening"
    arrival_time: str            # e.g. "Night", "Afternoon"
    stops: Literal["zero", "one", "two_or_more"]
    travel_class: Literal["Economy", "Business"]
    duration: float              # hours
    days_left: int               # days before departure


class PredictResponse(BaseModel):
    predicted_price: float


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Map the string stops value back to the numeric value 0/1/2 used during training
    stops_mapping = {"zero": 0, "one": 1, "two_or_more": 2}
    stops_value = stops_mapping[req.stops]

    # Build the route feature exactly as in training: "source_city-destination_city"
    route = f"{req.source_city}-{req.destination_city}"

    # Construct X with the same columns as in the training phase
    data = pd.DataFrame(
        [
            {
                "airline": req.airline,
                "departure_time": req.departure_time,
                "stops": stops_value,
                "arrival_time": req.arrival_time,
                "class": req.travel_class,
                "duration": float(req.duration),
                "days_left": float(req.days_left),
                "route": route,
            }
        ]
    )

    # The model predicts log_price, so we convert it back to price
    pred_log = model.predict(data)[0]
    pred_price = float(np.expm1(pred_log))

    return PredictResponse(predicted_price=pred_price)
