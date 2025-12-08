# api.py
#
# FastAPI backend for the flight price project.
# Exposes:
#   - GET  /health        : health check
#   - POST /predict       : point-wise price prediction for a given days_left
#   - POST /booking_plan  : price curve over a window of days_left and optimal booking time
#   - GET  /demo          : HTML demo page (uses templates/demo.html)

from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# -------------------------------------------------------------------
# Paths and model loading
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

app = FastAPI(
    title="Flight Price Prediction API",
    description="API for predicting flight prices and optimal booking time.",
    version="1.0.0",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

STOPS_MAP = {
    "zero": 0,
    "one": 1,
    "two_or_more": 2,
}

# We try to load the model at startup so that errors are visible early.
try:
    model = joblib.load(MODEL_PATH)
    model_loading_error: str | None = None
except Exception as e:  # noqa: BLE001
    model = None
    model_loading_error = str(e)


# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------

class FlightRequest(BaseModel):
    """Request schema for /predict (includes days_left)."""

    airline: str = Field(..., description="Airline name, e.g. 'IndiGo'")
    source_city: str = Field(..., description="Source city, e.g. 'Delhi'")
    destination_city: str = Field(..., description="Destination city, e.g. 'Mumbai'")
    departure_time: str = Field(..., description="Departure time slot, e.g. 'Morning'")
    arrival_time: str = Field(..., description="Arrival time slot, e.g. 'Night'")
    stops: str = Field(
        ...,
        description="Number of stops in text form: 'zero', 'one', or 'two_or_more'",
    )
    travel_class: str = Field(..., description="Travel class, e.g. 'Economy'")
    duration: float = Field(..., description="Flight duration in hours")
    days_left: int = Field(..., description="How many days before departure the ticket is bought")


class FlightResponse(BaseModel):
    predicted_price: float


class BookingRequest(BaseModel):
    """Request schema for /booking_plan (no days_left; the API scans over a window)."""

    airline: str
    source_city: str
    destination_city: str
    departure_time: str
    arrival_time: str
    stops: str  # "zero" | "one" | "two_or_more"
    travel_class: str
    duration: float


class BookingPlanPoint(BaseModel):
    days_left: int
    predicted_price: float


class BookingPlanResponse(BaseModel):
    best_days_left: int
    best_price: float
    curve: List[BookingPlanPoint]


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def ensure_model_loaded():
    """Raise HTTP 500 if the model failed to load."""
    if model is None:
        msg = "Model is not loaded. Please check that models/model.joblib exists."
        if model_loading_error:
            msg += f" Loading error: {model_loading_error}"
        raise HTTPException(status_code=500, detail=msg)


def build_single_df(req: FlightRequest) -> pd.DataFrame:
    """Build a single-row DataFrame from a FlightRequest."""
    stops_encoded = STOPS_MAP.get(req.stops, 0)
    row = {
        "airline": req.airline,
        "source_city": req.source_city,
        "destination_city": req.destination_city,
        "departure_time": req.departure_time,
        "arrival_time": req.arrival_time,
        "class": req.travel_class,
        "duration": req.duration,
        "days_left": req.days_left,
        "stops_encoded": stops_encoded,
        "route": f"{req.source_city}-{req.destination_city}",
    }
    return pd.DataFrame([row])


def build_curve_df(
    req: BookingRequest,
    days_min: int = 1,
    days_max: int = 60,
) -> pd.DataFrame:
    """Given a flight profile, build a DataFrame over a window of days_left."""
    rows = []
    stops_encoded = STOPS_MAP.get(req.stops, 0)

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
                "stops_encoded": stops_encoded,
                "route": f"{req.source_city}-{req.destination_city}",
            }
        )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=FlightResponse)
def predict_price(req: FlightRequest):
    """
    Point-wise prediction: given a specific days_left, predict ticket price.

    This is useful if we only want to answer:
    "If I buy the ticket N days before departure, what is the expected price?"
    """
    ensure_model_loaded()

    df_input = build_single_df(req)
    try:
        log_pred = model.predict(df_input)[0]
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}") from e

    price = float(np.expm1(log_pred))  # inverse of log1p(price)
    return FlightResponse(predicted_price=price)


@app.post("/booking_plan", response_model=BookingPlanResponse)
def booking_plan(req: BookingRequest):
    """
    For a fixed flight profile, scan days_left over a window [1, 60] and return:
    - the full price curve, and
    - the recommended booking time (days_left with minimum predicted price).
    """
    ensure_model_loaded()

    df_curve = build_curve_df(req, days_min=1, days_max=60)

    try:
        log_pred = model.predict(df_curve)  # shape: (N,)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}") from e

    prices = np.expm1(log_pred)  # back to original price scale

    # Find minimum
    idx_min = int(np.argmin(prices))
    best_days_left = int(df_curve.loc[idx_min, "days_left"])
    best_price = float(prices[idx_min])

    curve_points = [
        BookingPlanPoint(days_left=int(d), predicted_price=float(p))
        for d, p in zip(df_curve["days_left"].tolist(), prices.tolist())
    ]

    return BookingPlanResponse(
        best_days_left=best_days_left,
        best_price=best_price,
        curve=curve_points,
    )


@app.get("/demo")
def demo_page(request: Request):
    """
    Render the HTML demo page.

    The template 'demo.html' should be located in the 'templates' folder
    and will call the /booking_plan endpoint from JavaScript.
    """
    return templates.TemplateResponse("demo.html", {"request": request})


@app.get("/")
def root():
    """Simple root endpoint to point users to the docs and demo."""
    return JSONResponse(
        {
            "message": "Flight Price Prediction API",
            "docs": "/docs",
            "demo": "/demo",
        }
    )
