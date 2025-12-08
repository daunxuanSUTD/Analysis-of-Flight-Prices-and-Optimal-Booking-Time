# api.py  - English version

from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(title="Flight Price Booking Planner")

# Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

# Load trained model (pipeline: preprocessing + regressor)
MODEL_PATH = "models/model.joblib"
model = joblib.load(MODEL_PATH)

# Mapping from text stops to numeric encoding
STOPS_MAP = {
    "zero": 0,
    "one": 1,
    "two_or_more": 2,
}


# ---------- Health & basic routes ----------


@app.get("/health")
def health() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.get("/")
def index() -> RedirectResponse:
    """Redirect root URL to the demo page."""
    return RedirectResponse(url="/demo")


@app.get("/demo")
def demo_page(request: Request):
    """Render the front-end demo page."""
    return templates.TemplateResponse("demo.html", {"request": request})


# ---------- Single-point prediction (optional, kept for completeness) ----------


class PredictRequest(BaseModel):
    airline: str
    source_city: str
    destination_city: str
    departure_time: str
    arrival_time: str
    stops: str  # "zero" | "one" | "two_or_more"
    travel_class: str  # "Economy" | "Business" | "First"
    duration: float
    days_left: int


class PredictResponse(BaseModel):
    predicted_price: float


@app.post("/predict", response_model=PredictResponse)
def predict_price(req: PredictRequest) -> PredictResponse:
    """
    Single-point prediction:
    Given a specific configuration INCLUDING days_left,
    return the predicted ticket price.
    """
    df = pd.DataFrame(
        [
            {
                "airline": req.airline,
                "source_city": req.source_city,
                "destination_city": req.destination_city,
                "departure_time": req.departure_time,
                "arrival_time": req.arrival_time,
                "class": req.travel_class,
                "duration": req.duration,
                "days_left": req.days_left,
                # both raw and encoded stops are provided
                "stops": req.stops,
                "stops_encoded": STOPS_MAP.get(req.stops, 0),
                "route": f"{req.source_city}-{req.destination_city}",
            }
        ]
    )

    # Model predicts log_price; convert back to original price scale
    log_pred = model.predict(df)[0]
    price = float(np.expm1(log_pred))

    return PredictResponse(predicted_price=price)


# ---------- Optimal booking plan (used by the demo UI) ----------


class BookingRequest(BaseModel):
    airline: str
    source_city: str
    destination_city: str
    departure_time: str
    arrival_time: str
    stops: str  # "zero" | "one" | "two_or_more"
    travel_class: str  # "Economy" | "Business" | "First"
    duration: float  # in hours


class BookingPlanPoint(BaseModel):
    days_left: int
    predicted_price: float


class BookingPlanResponse(BaseModel):
    best_days_left: int
    best_price: float
    curve: List[BookingPlanPoint]


def build_curve_df(
    req: BookingRequest, days_min: int = 1, days_max: int = 60
) -> pd.DataFrame:
    """
    Build a DataFrame where all flight features are fixed,
    and only `days_left` varies from days_min to days_max.
    """
    rows = []
    for d in range(days_min, days_max + 1):
        rows.append(
            {
                "airline": req.airline,
                "source_city": req.source_city,
                "destination_city": req.destination_city,
                "departure_time": req.departure_time,
                "arrival_time": req.arrival_time,
                "class": req.travel_class,
                "duration": req.duration,
                "days_left": d,
                # provide both columns because the trained pipeline expects them
                "stops": req.stops,
                "stops_encoded": STOPS_MAP.get(req.stops, 0),
                "route": f"{req.source_city}-{req.destination_city}",
            }
        )
    return pd.DataFrame(rows)


@app.post("/booking_plan", response_model=BookingPlanResponse)
def booking_plan(req: BookingRequest) -> BookingPlanResponse:
    """
    Main endpoint used by the front-end demo.

    For a given flight configuration (airline, route, class, etc.),
    we:
    - Enumerate days_left from 1 to 60
    - Predict price for each day
    - Build a price curve
    - Find the day with the lowest predicted price
    """
    # 1) Build curve DataFrame
    df_curve = build_curve_df(req, days_min=1, days_max=60)

    # 2) Predict log_price and convert to price
    log_pred = model.predict(df_curve)  # shape: (N,)
    prices = np.expm1(log_pred)

    # 3) Find the minimum predicted price
    idx_min = int(np.argmin(prices))
    best_days_left = int(df_curve.loc[idx_min, "days_left"])
    best_price = float(prices[idx_min])

    # 4) Build response curve
    curve: List[BookingPlanPoint] = [
        BookingPlanPoint(days_left=int(d), predicted_price=float(p))
        for d, p in zip(df_curve["days_left"].tolist(), prices.tolist())
    ]

    return BookingPlanResponse(
        best_days_left=best_days_left,
        best_price=best_price,
        curve=curve,
    )
