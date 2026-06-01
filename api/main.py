"""
FastAPI Backend for Traffic Intelligence Dashboard
Serves predictions, analytics, and insights.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import (
    MODELS_DIR, TRAIN_FILE, TEST_FILE,
    WEATHER_SEVERITY_MAP, ROAD_TYPE_MAP, TARGET
)

logger = logging.getLogger(__name__)

# ─── Global State ────────────────────────────────────────────────────────────
app_state = {
    "models": {},
    "feature_engineer": None,
    "feature_cols": None,
    "ensemble_config": None,
    "train_data": None,
    "analytics_cache": {},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and data on startup."""
    logger.info("Loading models and data...")

    try:
        # Load feature engineer
        fe_path = MODELS_DIR / "feature_engineer.joblib"
        if fe_path.exists():
            app_state["feature_engineer"] = joblib.load(fe_path)

        # Load feature columns
        fc_path = MODELS_DIR / "feature_columns.joblib"
        if fc_path.exists():
            app_state["feature_cols"] = joblib.load(fc_path)

        # Load models
        for name, fname in [("CatBoost", "catboost_tuned.joblib"),
                             ("LightGBM", "lightgbm_tuned.joblib"),
                             ("XGBoost", "xgboost_tuned.joblib")]:
            path = MODELS_DIR / fname
            if path.exists():
                app_state["models"][name] = joblib.load(path)
                logger.info(f"  Loaded {name}")

        # Load ensemble config
        config_path = MODELS_DIR / "ensemble_config.joblib"
        if config_path.exists():
            app_state["ensemble_config"] = joblib.load(config_path)

        # Load and prepare training data for analytics
        if TRAIN_FILE.exists():
            train_df = pd.read_csv(TRAIN_FILE)
            # Parse timestamp
            parts = train_df["timestamp"].astype(str).str.split(":", expand=True)
            train_df["hour"] = parts[0].astype(int)
            train_df["weekday"] = train_df["day"] % 7
            app_state["train_data"] = train_df
            _precompute_analytics(train_df)

        logger.info("Startup complete.")
    except Exception as e:
        logger.error(f"Startup error: {e}")

    yield

    logger.info("Shutting down.")


def _precompute_analytics(df: pd.DataFrame):
    """Precompute analytics for fast API responses."""
    cache = app_state["analytics_cache"]

    # Hourly demand
    hourly = df.groupby("hour")[TARGET].agg(["mean", "std", "count"]).reset_index()
    cache["hourly_demand"] = hourly.to_dict(orient="records")

    # Weather impact
    weather_valid = df.dropna(subset=["Weather"])
    weather_stats = weather_valid.groupby("Weather")[TARGET].agg(
        ["mean", "median", "std", "count"]
    ).reset_index()
    cache["weather_impact"] = weather_stats.to_dict(orient="records")

    # Road capacity
    road_valid = df.dropna(subset=["RoadType"])
    road_stats = road_valid.groupby("RoadType")[TARGET].agg(
        ["mean", "median", "std", "count"]
    ).reset_index()
    cache["road_capacity"] = road_stats.to_dict(orient="records")

    # Lane analysis
    lane_stats = df.groupby("NumberofLanes")[TARGET].agg(
        ["mean", "std", "count"]
    ).reset_index()
    cache["lane_analysis"] = lane_stats.to_dict(orient="records")

    # Weekday demand
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_stats = df.groupby("weekday")[TARGET].agg(["mean", "std"]).reset_index()
    weekday_stats["name"] = weekday_stats["weekday"].map(lambda x: weekday_names[x])
    cache["weekday_demand"] = weekday_stats.to_dict(orient="records")

    # Peak hours detection
    hourly_sorted = hourly.sort_values("mean", ascending=False)
    peak_hours = hourly_sorted.head(5)["hour"].tolist()
    off_peak = hourly_sorted.tail(5)["hour"].tolist()
    cache["peak_hours"] = {
        "peak": peak_hours,
        "off_peak": off_peak,
        "peak_avg_demand": float(hourly_sorted.head(5)["mean"].mean()),
        "off_peak_avg_demand": float(hourly_sorted.tail(5)["mean"].mean()),
    }

    # Heatmap data (geohash-based)
    from src.features.engineering import FeatureEngineer
    geo_demand = df.groupby("geohash")[TARGET].mean().reset_index()
    geo_demand["latitude"], geo_demand["longitude"] = zip(
        *geo_demand["geohash"].apply(FeatureEngineer._decode_geohash)
    )
    cache["heatmap"] = geo_demand.to_dict(orient="records")


# ─── App Setup ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Traffic Intelligence API",
    description="AI-powered traffic demand prediction and analytics",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ─────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    geohash: str
    day: int
    timestamp: str
    RoadType: Optional[str] = "Residential"
    NumberofLanes: int = 2
    LargeVehicles: str = "Not Allowed"
    Landmarks: str = "No"
    Temperature: Optional[float] = 20.0
    Weather: Optional[str] = "Sunny"


class PredictionResponse(BaseModel):
    demand: float
    model_used: str


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": list(app_state["models"].keys()),
        "data_loaded": app_state["train_data"] is not None,
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict traffic demand for given conditions."""
    if not app_state["models"]:
        raise HTTPException(500, "No models loaded")

    if app_state["feature_engineer"] is None:
        raise HTTPException(500, "Feature engineer not loaded")

    # Create single-row DataFrame
    input_df = pd.DataFrame([{
        "Index": 0,
        "geohash": request.geohash,
        "day": request.day,
        "timestamp": request.timestamp,
        "RoadType": request.RoadType,
        "NumberofLanes": request.NumberofLanes,
        "LargeVehicles": request.LargeVehicles,
        "Landmarks": request.Landmarks,
        "Temperature": request.Temperature,
        "Weather": request.Weather,
    }])

    # Transform
    fe = app_state["feature_engineer"]
    transformed = fe.transform(input_df)
    feature_cols = app_state["feature_cols"]
    X = transformed[feature_cols].fillna(0).values

    # Predict with all models and average
    predictions = []
    for name, model in app_state["models"].items():
        pred = float(np.clip(model.predict(X)[0], 0, None))
        predictions.append(pred)

    avg_pred = float(np.mean(predictions))

    return PredictionResponse(
        demand=round(avg_pred, 6),
        model_used="ensemble_avg"
    )


@app.get("/api/analytics/demand-by-hour")
async def demand_by_hour():
    """Get average demand by hour of day."""
    return app_state["analytics_cache"].get("hourly_demand", [])


@app.get("/api/analytics/demand-by-weather")
async def demand_by_weather():
    """Get demand statistics by weather condition."""
    return app_state["analytics_cache"].get("weather_impact", [])


@app.get("/api/analytics/demand-by-weekday")
async def demand_by_weekday():
    """Get demand by day of week."""
    return app_state["analytics_cache"].get("weekday_demand", [])


@app.get("/api/analytics/heatmap")
async def heatmap():
    """Get geospatial demand data for heatmap."""
    return app_state["analytics_cache"].get("heatmap", [])


@app.get("/api/analytics/road-capacity")
async def road_capacity():
    """Get road capacity and utilization data."""
    return {
        "by_road_type": app_state["analytics_cache"].get("road_capacity", []),
        "by_lanes": app_state["analytics_cache"].get("lane_analysis", []),
    }


@app.get("/api/analytics/peak-hours")
async def peak_hours():
    """Get peak hour detection results."""
    return app_state["analytics_cache"].get("peak_hours", {})


@app.get("/api/forecast")
async def forecast():
    """Get demand forecast data (hourly pattern)."""
    cache = app_state["analytics_cache"]
    hourly = cache.get("hourly_demand", [])

    # Create 24-hour forecast based on historical patterns
    forecast_data = []
    for entry in hourly:
        forecast_data.append({
            "hour": entry["hour"],
            "predicted_demand": round(entry["mean"], 4),
            "confidence_low": round(max(0, entry["mean"] - 1.96 * entry.get("std", 0)), 4),
            "confidence_high": round(entry["mean"] + 1.96 * entry.get("std", 0), 4),
        })

    return forecast_data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
