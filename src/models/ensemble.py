"""
Phase 7: Ensemble Learning
Builds weighted and stacking ensembles for maximum prediction accuracy.
"""

import numpy as np
import logging
import joblib
from typing import Dict, List

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.optimize import minimize

from src.config import CV_FOLDS, RANDOM_SEED, MODELS_DIR

logger = logging.getLogger(__name__)


class EnsembleBuilder:
    """Builds and evaluates ensemble models."""

    def __init__(self):
        self.models = {}
        self.ensemble_weights = None
        self.stacking_meta_model = None
        self.best_ensemble_type = None
        self.best_ensemble_r2 = -np.inf

    def set_models(self, models: Dict):
        """Set the base models for ensembling."""
        self.models = models
        logger.info(f"  Ensemble base models: {list(models.keys())}")

    def _weighted_prediction(self, predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute weighted average prediction."""
        return np.average(predictions, weights=weights, axis=0)

    def build_weighted_ensemble(self, X: np.ndarray, y: np.ndarray) -> float:
        """Optimize weights for weighted ensemble."""
        logger.info("\n  Building weighted ensemble...")

        model_names = list(self.models.keys())
        n_models = len(model_names)

        if n_models < 2:
            logger.warning("  Need at least 2 models for ensemble.")
            return -np.inf

        # Get OOF predictions for weight optimization
        kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        oof_predictions = np.zeros((len(y), n_models))

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            for i, name in enumerate(model_names):
                model = self._clone_model(name)
                if name == "CatBoost":
                    model.fit(X_train, y_train, eval_set=(X_val, y[val_idx]), verbose=0)
                elif name == "LightGBM":
                    import lightgbm as lgb
                    model.fit(X_train, y_train, eval_set=[(X_val, y[val_idx])],
                              callbacks=[lgb.early_stopping(50, verbose=False),
                                         lgb.log_evaluation(0)])
                elif name == "XGBoost":
                    model.fit(X_train, y_train, eval_set=[(X_val, y[val_idx])], verbose=False)
                else:
                    model.fit(X_train, y_train)

                oof_predictions[val_idx, i] = np.clip(model.predict(X_val), 0, None)

        # Optimize weights
        def neg_r2(weights):
            weights = np.abs(weights) / np.sum(np.abs(weights))
            pred = self._weighted_prediction(oof_predictions.T, weights)
            return -r2_score(y, pred)

        initial_weights = np.ones(n_models) / n_models
        result = minimize(neg_r2, initial_weights, method="Nelder-Mead",
                          options={"maxiter": 1000})
        optimal_weights = np.abs(result.x) / np.sum(np.abs(result.x))

        self.ensemble_weights = dict(zip(model_names, optimal_weights))
        weighted_r2 = -result.fun

        logger.info(f"  Weighted ensemble R²: {weighted_r2:.4f}")
        for name, w in self.ensemble_weights.items():
            logger.info(f"    {name}: {w:.4f}")

        return weighted_r2

    def build_stacking_ensemble(self, X: np.ndarray, y: np.ndarray) -> float:
        """Build stacking ensemble with Ridge meta-learner."""
        logger.info("\n  Building stacking ensemble...")

        model_names = list(self.models.keys())
        n_models = len(model_names)

        if n_models < 2:
            logger.warning("  Need at least 2 models for stacking.")
            return -np.inf

        # Generate out-of-fold predictions (Level 1)
        kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        oof_predictions = np.zeros((len(y), n_models))

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            for i, name in enumerate(model_names):
                model = self._clone_model(name)
                if name == "CatBoost":
                    model.fit(X_train, y_train, eval_set=(X_val, y[val_idx]), verbose=0)
                elif name == "LightGBM":
                    import lightgbm as lgb
                    model.fit(X_train, y_train, eval_set=[(X_val, y[val_idx])],
                              callbacks=[lgb.early_stopping(50, verbose=False),
                                         lgb.log_evaluation(0)])
                elif name == "XGBoost":
                    model.fit(X_train, y_train, eval_set=[(X_val, y[val_idx])], verbose=False)
                else:
                    model.fit(X_train, y_train)

                oof_predictions[val_idx, i] = np.clip(model.predict(X_val), 0, None)

        # Level 2: Train meta-learner on OOF predictions
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(oof_predictions, y)
        meta_predictions = np.clip(meta_model.predict(oof_predictions), 0, None)
        stacking_r2 = r2_score(y, meta_predictions)

        self.stacking_meta_model = meta_model
        logger.info(f"  Stacking ensemble R²: {stacking_r2:.4f}")
        logger.info(f"  Meta-learner coefficients: {dict(zip(model_names, meta_model.coef_.round(4)))}")

        return stacking_r2

    def _clone_model(self, name: str):
        """Create a fresh copy of a model with same parameters."""
        model = self.models[name]

        if name == "CatBoost":
            from catboost import CatBoostRegressor
            # Only use safe, user-settable params — get_all_params() includes internal ones
            safe_keys = {
                "iterations", "depth", "learning_rate", "l2_leaf_reg",
                "subsample", "random_seed", "verbose", "early_stopping_rounds",
                "min_child_samples", "border_count", "thread_count",
            }
            all_params = model.get_params()
            params = {k: v for k, v in all_params.items() if k in safe_keys and v is not None}
            params.setdefault("verbose", 0)
            params.setdefault("early_stopping_rounds", 50)
            return CatBoostRegressor(**params)
        elif name == "LightGBM":
            import lightgbm as lgb
            params = model.get_params()
            return lgb.LGBMRegressor(**params)
        elif name == "XGBoost":
            import xgboost as xgb
            params = model.get_params()
            # Ensure early_stopping_rounds is set for eval_set usage
            params["early_stopping_rounds"] = params.get("early_stopping_rounds", 50)
            return xgb.XGBRegressor(**params)
        else:
            from sklearn.base import clone
            return clone(model)

    def select_best_ensemble(self, X: np.ndarray, y: np.ndarray):
        """Build both ensembles and select the best one."""
        logger.info("=" * 60)
        logger.info("PHASE 7: ENSEMBLE LEARNING")
        logger.info("=" * 60)

        weighted_r2 = self.build_weighted_ensemble(X, y)
        stacking_r2 = self.build_stacking_ensemble(X, y)

        if weighted_r2 >= stacking_r2:
            self.best_ensemble_type = "weighted"
            self.best_ensemble_r2 = weighted_r2
            logger.info(f"\n  Best ensemble: WEIGHTED (R²={weighted_r2:.4f})")
        else:
            self.best_ensemble_type = "stacking"
            self.best_ensemble_r2 = stacking_r2
            logger.info(f"\n  Best ensemble: STACKING (R²={stacking_r2:.4f})")

        # Save ensemble config
        ensemble_config = {
            "best_type": self.best_ensemble_type,
            "best_r2": self.best_ensemble_r2,
            "weighted_r2": weighted_r2,
            "stacking_r2": stacking_r2,
            "weights": self.ensemble_weights,
        }

        if self.stacking_meta_model is not None:
            joblib.dump(self.stacking_meta_model, MODELS_DIR / "stacking_meta_model.joblib")

        joblib.dump(ensemble_config, MODELS_DIR / "ensemble_config.joblib")
        logger.info("Phase 7 complete.")

        return self.best_ensemble_type

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the best ensemble."""
        model_names = list(self.models.keys())
        predictions = np.column_stack([
            np.clip(self.models[name].predict(X), 0, None)
            for name in model_names
        ])

        if self.best_ensemble_type == "weighted":
            weights = [self.ensemble_weights[name] for name in model_names]
            return np.clip(self._weighted_prediction(predictions.T, weights), 0, None)
        else:
            return np.clip(self.stacking_meta_model.predict(predictions), 0, None)
