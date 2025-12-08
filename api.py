from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# -----------------------------
# Paths & model loading
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"  
TEMPLATES_DIR = PROJECT_ROOT / "templates"

app = FastAPI(
    title="Flight Price Prediction API",
    description="Predict flight ticket prices based on a trained log-price model.",
    version="1.0.0",
)

# Jinja2 templates (for the light demo page)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

try:
    # Load the full pipeline (preprocessing + model)
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    # 明确报错，方便调试
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. "
                       f"Please run train.py to generate model.joblib first.")


# -----------------------------
# Pydantic models
# -----------------------------
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


# -----------------------------
# API endpoints
# -----------------------------
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

    try:
        # The model predicts log_price, so we convert it back to price
        pred_log = model.predict(data)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    pred_price = float(np.expm1(pred_log))

    return PredictResponse(predicted_price=pred_price)


# -----------------------------
# Light demo pages
# -----------------------------
@app.get("/", response_class=HTMLResponse)
@app.get("/demo", response_class=HTMLResponse)
def demo_page(request: Request):
    """
    Simple HTML demo page that lets users input flight info
    and calls /predict in the background.
    """
    return templates.TemplateResponse("demo.html", {"request": request})


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}
