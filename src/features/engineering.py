"""
Phase 3: Feature Engineering
Transforms raw features into ML-ready features for traffic demand prediction.
"""

import pandas as pd
import numpy as np
import logging
import warnings
from typing import Tuple

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from src.config import (
    ROAD_TYPE_MAP, WEATHER_SEVERITY_MAP, RANDOM_SEED,
    KMEANS_CLUSTERS, TARGET,
    MORNING_PEAK_START, MORNING_PEAK_END,
    EVENING_PEAK_START, EVENING_PEAK_END,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Transforms raw data into ML-ready features."""

    def __init__(self):
        self.label_encoders = {}
        self.kmeans_models = {}
        self.geo_frequency_map = None
        self.feature_columns = []
        self._is_fitted = False

    def _parse_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract hour and minute from timestamp string."""
        parts = df["timestamp"].astype(str).str.split(":", expand=True)
        df["hour"] = parts[0].astype(int)
        df["minute"] = parts[1].astype(int) if parts.shape[1] > 1 else 0
        return df

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from timestamp and day."""
        logger.info("  Creating time features...")

        df = self._parse_timestamp(df)

        # Weekday from day column (day modulo 7)
        df["weekday"] = df["day"] % 7

        # Week and month estimates
        df["week"] = df["day"] // 7
        df["month"] = ((df["day"] // 30) % 12 + 1).astype(int)
        df["quarter"] = ((df["month"] - 1) // 3 + 1).astype(int)

        # Binary flags
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)
        df["is_morning_peak"] = ((df["hour"] >= MORNING_PEAK_START) &
                                  (df["hour"] <= MORNING_PEAK_END)).astype(int)
        df["is_evening_peak"] = ((df["hour"] >= EVENING_PEAK_START) &
                                  (df["hour"] <= EVENING_PEAK_END)).astype(int)
        df["is_peak_hour"] = (df["is_morning_peak"] | df["is_evening_peak"]).astype(int)

        return df

    def _create_cyclic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical sin/cos encodings for time features."""
        logger.info("  Creating cyclic features...")

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

        return df

    def _create_geospatial_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Decode geohash and create spatial features."""
        logger.info("  Creating geospatial features...")

        # Decode geohash to lat/lon using a simple decoder (avoid pygeohash dependency issues)
        df["latitude"], df["longitude"] = zip(*df["geohash"].apply(self._decode_geohash))

        # Geo frequency encoding
        if fit:
            self.geo_frequency_map = df["geohash"].value_counts().to_dict()

        df["geo_frequency"] = df["geohash"].map(self.geo_frequency_map).fillna(1)

        # KMeans clustering on lat/lon
        coords = df[["latitude", "longitude"]].values

        for k in KMEANS_CLUSTERS:
            col_name = f"region_cluster_{k}"
            if fit:
                kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
                kmeans.fit(coords)
                self.kmeans_models[k] = kmeans
            df[col_name] = self.kmeans_models[k].predict(coords)

        return df

    @staticmethod
    def _decode_geohash(geohash_str: str) -> Tuple[float, float]:
        """Decode a geohash string into latitude and longitude."""
        base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
        bits = []
        for char in geohash_str:
            idx = base32.index(char)
            bits.extend([(idx >> (4 - i)) & 1 for i in range(5)])

        lat_bits = bits[1::2]
        lon_bits = bits[0::2]

        def decode_range(bit_list, min_val, max_val):
            for bit in bit_list:
                mid = (min_val + max_val) / 2
                if bit:
                    min_val = mid
                else:
                    max_val = mid
            return (min_val + max_val) / 2

        lat = decode_range(lat_bits, -90, 90)
        lon = decode_range(lon_bits, -180, 180)
        return lat, lon

    def _handle_missing_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Impute missing values before feature creation."""
        logger.info("  Handling missing values...")

        # RoadType: fill with mode
        if fit:
            self._roadtype_mode = df["RoadType"].mode()[0] if not df["RoadType"].mode().empty else "Residential"
        df["RoadType"] = df["RoadType"].fillna(self._roadtype_mode)

        # Weather: fill with mode
        if fit:
            self._weather_mode = df["Weather"].mode()[0] if not df["Weather"].mode().empty else "Sunny"
        df["Weather"] = df["Weather"].fillna(self._weather_mode)

        # Temperature: fill with median
        if fit:
            self._temp_median = df["Temperature"].median()
        df["Temperature"] = df["Temperature"].fillna(self._temp_median)

        return df

    def _create_infrastructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create road infrastructure features."""
        logger.info("  Creating infrastructure features...")

        # Road type ordinal
        df["road_type_encoded"] = df["RoadType"].map(ROAD_TYPE_MAP).fillna(1).astype(int)

        # Large vehicles binary
        df["large_vehicles_allowed"] = (df["LargeVehicles"] == "Allowed").astype(int)

        # Landmarks binary
        df["has_landmarks"] = (df["Landmarks"] == "Yes").astype(int)

        # Road capacity score
        df["road_capacity_score"] = (
            df["road_type_encoded"] *
            df["NumberofLanes"] *
            (1 + 0.5 * df["large_vehicles_allowed"])
        )

        # Interaction features (only those not dependent on weather features)
        df["lanes_x_temperature"] = df["NumberofLanes"] * df["Temperature"]

        return df

    def _create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weather-related features."""
        logger.info("  Creating weather features...")

        # Weather severity score
        df["weather_severity_score"] = df["Weather"].map(WEATHER_SEVERITY_MAP).fillna(1).astype(int)

        # Temperature bins (5 quantile-based bins)
        df["temperature_bin"] = pd.qcut(df["Temperature"], q=5, labels=False, duplicates="drop")
        df["temperature_bin"] = df["temperature_bin"].fillna(2).astype(int)

        # Adverse weather flag
        df["adverse_weather"] = (df["Weather"].isin(["Rainy", "Snowy"])).astype(int)

        # Adverse weather x lanes (doesn't depend on infrastructure)
        df["adverse_weather_x_lanes"] = df["adverse_weather"] * df["NumberofLanes"]

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cross-domain interaction features (requires infrastructure + weather)."""
        logger.info("  Creating interaction features...")

        df["weather_x_capacity"] = df["weather_severity_score"] * df["road_capacity_score"]
        df["lanes_x_weather_severity"] = df["NumberofLanes"] * df["weather_severity_score"]
        df["roadtype_x_weather"] = df["road_type_encoded"] * df["weather_severity_score"]

        return df

    def _create_landmark_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create landmark-based features."""
        logger.info("  Creating landmark features...")

        df["landmark_density_score"] = df["has_landmarks"] * df["geo_frequency"]
        df["landmark_lane_interaction"] = df["has_landmarks"] * df["NumberofLanes"]

        return df

    def _get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get list of feature columns, excluding identifiers and target."""
        exclude = ["Index", "geohash", "day", "timestamp", TARGET,
                    "RoadType", "Weather", "LargeVehicles", "Landmarks",
                    "_hour", "_minute", "_weekday", "_month", "_geo_prefix"]
        return [c for c in df.columns if c not in exclude and not c.startswith("_")]

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and transform it."""
        logger.info("=" * 60)
        logger.info("PHASE 3: FEATURE ENGINEERING (fit_transform)")
        logger.info("=" * 60)

        df = train_df.copy()

        # Step 1: Missing values
        df = self._handle_missing_values(df, fit=True)

        # Step 2: Time features
        df = self._create_time_features(df)

        # Step 3: Cyclic features
        df = self._create_cyclic_features(df)

        # Step 4: Geospatial features
        df = self._create_geospatial_features(df, fit=True)

        # Step 5: Infrastructure features (before weather — weather interactions need road_capacity_score)
        df = self._create_infrastructure_features(df)

        # Step 6: Weather features
        df = self._create_weather_features(df)

        # Step 7: Cross-domain interactions (needs both infrastructure + weather)
        df = self._create_interaction_features(df)

        # Step 8: Landmark features
        df = self._create_landmark_features(df)

        # Store feature columns
        self.feature_columns = self._get_feature_columns(df)
        self._is_fitted = True

        logger.info(f"\n  Total features created: {len(self.feature_columns)}")
        logger.info("Phase 3 complete (fit_transform).")
        return df

    def transform(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Transform test data using fitted parameters."""
        assert self._is_fitted, "Must call fit_transform() before transform()"

        logger.info("PHASE 3: FEATURE ENGINEERING (transform)")

        df = test_df.copy()

        df = self._handle_missing_values(df, fit=False)
        df = self._create_time_features(df)
        df = self._create_cyclic_features(df)
        df = self._create_geospatial_features(df, fit=False)
        df = self._create_infrastructure_features(df)
        df = self._create_weather_features(df)
        df = self._create_interaction_features(df)
        df = self._create_landmark_features(df)

        logger.info("Phase 3 complete (transform).")
        return df
