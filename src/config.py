"""
Global configuration for the Traffic Demand Prediction System.
Centralizes all paths, constants, and hyperparameter defaults.
"""

import os
from pathlib import Path

# ─── Project Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUT_DIR / "reports"
MODELS_DIR = OUTPUT_DIR / "models"
SUBMISSIONS_DIR = OUTPUT_DIR / "submissions"
SHAP_DIR = OUTPUT_DIR / "shap"

# Data files
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"

# Create output directories
for d in [REPORTS_DIR, MODELS_DIR, SUBMISSIONS_DIR, SHAP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Random Seed ─────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Schema Definitions ─────────────────────────────────────────────────────
TRAIN_COLUMNS = [
    "Index", "geohash", "day", "timestamp", "demand",
    "RoadType", "NumberofLanes", "LargeVehicles", "Landmarks",
    "Temperature", "Weather"
]
TEST_COLUMNS = [
    "Index", "geohash", "day", "timestamp",
    "RoadType", "NumberofLanes", "LargeVehicles", "Landmarks",
    "Temperature", "Weather"
]
TARGET = "demand"

# ─── Feature Groups ─────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = ["RoadType", "LargeVehicles", "Landmarks", "Weather"]
NUMERICAL_FEATURES = ["NumberofLanes", "Temperature"]
ID_FEATURES = ["Index", "geohash", "day", "timestamp"]

# ─── Road Type Mapping ──────────────────────────────────────────────────────
ROAD_TYPE_MAP = {
    "Residential": 1,
    "Street": 2,
    "Highway": 3,
}

# ─── Weather Severity Mapping ───────────────────────────────────────────────
WEATHER_SEVERITY_MAP = {
    "Sunny": 0,
    "Foggy": 1,
    "Rainy": 2,
    "Snowy": 3,
}

# ─── Model Defaults ─────────────────────────────────────────────────────────
CV_FOLDS = 5
OPTUNA_TRIALS = 100
OPTUNA_CV_FOLDS = 3

# ─── KMeans Cluster Sizes ───────────────────────────────────────────────────
KMEANS_CLUSTERS = [5, 10, 15]

# ─── Peak Hour Definitions ──────────────────────────────────────────────────
MORNING_PEAK_START = 7
MORNING_PEAK_END = 9
EVENING_PEAK_START = 17
EVENING_PEAK_END = 19
