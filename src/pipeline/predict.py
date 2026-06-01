"""
Phase 9: Prediction Pipeline
Reusable inference pipeline for generating submission files.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from typing import Optional

from src.config import (
    TEST_FILE, MODELS_DIR, SUBMISSIONS_DIR, TARGET
)
from src.features.engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """End-to-end inference pipeline for test data predictions."""

    def __init__(self):
        self.models = {}
        self.ensemble_config = None
        self.feature_engineer = None
        self.stacking_meta_model = None

    def load_models(self, model_dir: Optional[Path] = None):
        """Load saved models from disk."""
        model_dir = model_dir or MODELS_DIR
        logger.info("Loading models...")

        model_files = {
            "CatBoost": "catboost_tuned.joblib",
            "LightGBM": "lightgbm_tuned.joblib",
            "XGBoost": "xgboost_tuned.joblib",
        }

        for name, filename in model_files.items():
            path = model_dir / filename
            if path.exists():
                self.models[name] = joblib.load(path)
                logger.info(f"  Loaded: {name}")
            else:
                logger.warning(f"  Not found: {path}")

        # Load ensemble config
        config_path = model_dir / "ensemble_config.joblib"
        if config_path.exists():
            self.ensemble_config = joblib.load(config_path)
            logger.info(f"  Ensemble type: {self.ensemble_config.get('best_type', 'unknown')}")

        # Load stacking meta-model if needed
        meta_path = model_dir / "stacking_meta_model.joblib"
        if meta_path.exists():
            self.stacking_meta_model = joblib.load(meta_path)

    def load_test_data(self, test_path: Optional[Path] = None) -> pd.DataFrame:
        """Load and validate test data."""
        test_path = test_path or TEST_FILE
        logger.info(f"Loading test data from {test_path}...")
        test_df = pd.read_csv(test_path)
        logger.info(f"  Test shape: {test_df.shape}")
        return test_df

    def apply_transformations(self, test_df: pd.DataFrame,
                               feature_engineer: FeatureEngineer) -> pd.DataFrame:
        """Apply the same feature engineering pipeline as training."""
        logger.info("Applying feature transformations...")
        self.feature_engineer = feature_engineer
        transformed = feature_engineer.transform(test_df)
        logger.info(f"  Transformed shape: {transformed.shape}")
        return transformed

    def generate_predictions(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using ensemble."""
        logger.info("Generating predictions...")

        model_names = list(self.models.keys())

        if len(model_names) == 0:
            raise ValueError("No models loaded!")

        # Get predictions from each model
        predictions = np.column_stack([
            np.clip(self.models[name].predict(X), 0, None)
            for name in model_names
        ])

        # Apply ensemble
        if self.ensemble_config and len(model_names) > 1:
            ensemble_type = self.ensemble_config.get("best_type", "weighted")

            if ensemble_type == "weighted":
                weights = [self.ensemble_config["weights"].get(name, 1.0/len(model_names))
                           for name in model_names]
                weights = np.array(weights) / sum(weights)
                final_pred = np.average(predictions, weights=weights, axis=1)
                logger.info(f"  Using weighted ensemble: {dict(zip(model_names, weights.round(4)))}")
            else:
                if self.stacking_meta_model is not None:
                    final_pred = self.stacking_meta_model.predict(predictions)
                    logger.info("  Using stacking ensemble")
                else:
                    final_pred = np.mean(predictions, axis=1)
                    logger.info("  Stacking meta-model not found, using simple average")
        elif len(model_names) == 1:
            final_pred = predictions[:, 0]
            logger.info(f"  Using single model: {model_names[0]}")
        else:
            final_pred = np.mean(predictions, axis=1)
            logger.info("  Using simple average")

        final_pred = np.clip(final_pred, 0, None)
        logger.info(f"  Predictions: mean={final_pred.mean():.4f}, "
                     f"std={final_pred.std():.4f}, "
                     f"min={final_pred.min():.4f}, max={final_pred.max():.4f}")

        return final_pred

    def export_submission(self, test_df: pd.DataFrame, predictions: np.ndarray,
                          filename: str = "submission.csv") -> Path:
        """Export predictions in submission format."""
        logger.info("Exporting submission...")

        submission = pd.DataFrame({
            "Index": test_df["Index"].values,
            TARGET: predictions
        })

        output_path = SUBMISSIONS_DIR / filename
        submission.to_csv(output_path, index=False)

        logger.info(f"  Submission saved: {output_path}")
        logger.info(f"  Shape: {submission.shape}")
        logger.info(f"  Preview:\n{submission.head()}")

        return output_path

    def run(self, feature_engineer: FeatureEngineer,
            feature_cols: list,
            test_path: Optional[Path] = None,
            submission_name: str = "submission.csv") -> Path:
        """Execute the full prediction pipeline."""
        logger.info("=" * 60)
        logger.info("PHASE 9: PREDICTION PIPELINE")
        logger.info("=" * 60)

        # Step 1: Load models
        self.load_models()

        # Step 2: Load test data
        test_df = self.load_test_data(test_path)

        # Step 3: Apply transformations
        transformed = self.apply_transformations(test_df, feature_engineer)

        # Step 4: Prepare feature matrix
        X_test = transformed[feature_cols].fillna(0).values

        # Step 5: Generate predictions
        predictions = self.generate_predictions(X_test)

        # Step 6: Export submission
        output_path = self.export_submission(test_df, predictions, submission_name)

        logger.info("Phase 9 complete.")
        return output_path
