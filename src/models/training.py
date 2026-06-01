"""
Phase 5: Model Development
Trains baseline and advanced regression models with cross-validation.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from typing import Dict, List, Any

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.config import CV_FOLDS, RANDOM_SEED, MODELS_DIR, TARGET

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and evaluates multiple regression models."""

    def __init__(self):
        self.models = {}
        self.cv_results = {}
        self.best_model_name = None
        self.best_score = -np.inf

    def _get_models(self) -> Dict[str, Any]:
        """Initialize all models."""
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_leaf=5,
                random_state=RANDOM_SEED, n_jobs=-1
            ),
        }

        # Advanced models
        try:
            from catboost import CatBoostRegressor
            models["CatBoost"] = CatBoostRegressor(
                iterations=1000, depth=8, learning_rate=0.1,
                l2_leaf_reg=5, random_seed=RANDOM_SEED, verbose=0,
                early_stopping_rounds=50
            )
        except ImportError:
            logger.warning("CatBoost not installed")

        try:
            import lightgbm as lgb
            models["LightGBM"] = lgb.LGBMRegressor(
                n_estimators=1000, max_depth=8, learning_rate=0.1,
                num_leaves=63, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=5, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1
            )
        except ImportError:
            logger.warning("LightGBM not installed")

        try:
            import xgboost as xgb
            models["XGBoost"] = xgb.XGBRegressor(
                n_estimators=1000, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_SEED, verbosity=0, n_jobs=-1,
                early_stopping_rounds=50
            )
        except ImportError:
            logger.warning("XGBoost not installed")

        return models

    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       feature_cols: List[str]) -> pd.DataFrame:
        """Run k-fold cross-validation for all models."""
        logger.info("=" * 60)
        logger.info("PHASE 5: MODEL DEVELOPMENT")
        logger.info("=" * 60)

        models = self._get_models()
        kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        results = []

        for name, model in models.items():
            logger.info(f"\n  Training {name}...")
            fold_metrics = {"r2": [], "rmse": [], "mae": []}

            for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Handle early stopping for boosting models
                if name in ["CatBoost", "LightGBM", "XGBoost"]:
                    if name == "CatBoost":
                        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
                    elif name == "LightGBM":
                        model.fit(X_train, y_train,
                                  eval_set=[(X_val, y_val)],
                                  callbacks=[
                                      __import__("lightgbm").early_stopping(50, verbose=False),
                                      __import__("lightgbm").log_evaluation(0)
                                  ])
                    elif name == "XGBoost":
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model.fit(X_train, y_train)

                # Predict and evaluate
                y_pred = model.predict(X_val)
                y_pred = np.clip(y_pred, 0, None)  # Demand can't be negative

                r2 = r2_score(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                mae = mean_absolute_error(y_val, y_pred)

                fold_metrics["r2"].append(r2)
                fold_metrics["rmse"].append(rmse)
                fold_metrics["mae"].append(mae)

                logger.info(f"    Fold {fold+1}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

            # Aggregate
            mean_r2 = np.mean(fold_metrics["r2"])
            std_r2 = np.std(fold_metrics["r2"])
            mean_rmse = np.mean(fold_metrics["rmse"])
            mean_mae = np.mean(fold_metrics["mae"])

            results.append({
                "Model": name,
                "R² Mean": round(mean_r2, 4),
                "R² Std": round(std_r2, 4),
                "RMSE Mean": round(mean_rmse, 4),
                "MAE Mean": round(mean_mae, 4),
            })

            logger.info(f"  {name} CV: R²={mean_r2:.4f}±{std_r2:.4f}")

            # Track best model
            if mean_r2 > self.best_score:
                self.best_score = mean_r2
                self.best_model_name = name

            # Store trained model (retrain on full data later)
            self.models[name] = model

        self.cv_results = pd.DataFrame(results).sort_values("R² Mean", ascending=False)
        logger.info(f"\n  Best model: {self.best_model_name} (R²={self.best_score:.4f})")

        # Save CV results
        self.cv_results.to_csv(MODELS_DIR / "cv_results.csv", index=False)
        logger.info("Phase 5 complete.")

        return self.cv_results

    def train_final_models(self, X: np.ndarray, y: np.ndarray, feature_cols: List[str]):
        """Retrain top models on full training data (no early stopping)."""
        logger.info("\n  Retraining final models on full data...")

        # Create fresh models WITHOUT early_stopping_rounds (no eval_set for full training)
        final_models = {}

        try:
            from catboost import CatBoostRegressor
            final_models["CatBoost"] = CatBoostRegressor(
                iterations=1000, depth=8, learning_rate=0.1,
                l2_leaf_reg=5, random_seed=RANDOM_SEED, verbose=0
            )
        except ImportError:
            pass

        try:
            import lightgbm as lgb
            final_models["LightGBM"] = lgb.LGBMRegressor(
                n_estimators=1000, max_depth=8, learning_rate=0.1,
                num_leaves=63, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=5, random_state=RANDOM_SEED, verbose=-1, n_jobs=-1
            )
        except ImportError:
            pass

        try:
            import xgboost as xgb
            final_models["XGBoost"] = xgb.XGBRegressor(
                n_estimators=1000, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_SEED, verbosity=0, n_jobs=-1
            )
        except ImportError:
            pass

        for name, model in final_models.items():
            logger.info(f"    Training {name} on full data...")
            model.fit(X, y)
            self.models[name] = model

            # Save as _tuned.joblib (matches prediction pipeline expectations)
            model_path = MODELS_DIR / f"{name.lower()}_tuned.joblib"
            joblib.dump(model, model_path)
            logger.info(f"    Saved: {model_path}")

        return self.models

