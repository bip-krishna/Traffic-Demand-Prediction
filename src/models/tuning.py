"""
Phase 6: Hyperparameter Optimization
Uses Optuna for Bayesian hyperparameter search on CatBoost, LightGBM, XGBoost.
"""

import numpy as np
import logging
import json
import joblib
from typing import Dict, Any

import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from src.config import (
    OPTUNA_TRIALS, OPTUNA_CV_FOLDS, RANDOM_SEED,
    MODELS_DIR, REPORTS_DIR
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Optimizes hyperparameters for advanced models using Optuna."""

    def __init__(self, n_trials: int = OPTUNA_TRIALS):
        self.n_trials = n_trials
        self.best_params = {}
        self.best_models = {}
        self.studies = {}

    def _catboost_objective(self, trial, X, y):
        """Optuna objective for CatBoost."""
        from catboost import CatBoostRegressor

        params = {
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "iterations": trial.suggest_int("iterations", 500, 3000),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "random_seed": RANDOM_SEED,
            "verbose": 0,
            "early_stopping_rounds": 50,
        }

        kfold = KFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = []

        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)

            y_pred = np.clip(model.predict(X_val), 0, None)
            scores.append(r2_score(y_val, y_pred))

        return np.mean(scores)

    def _lightgbm_objective(self, trial, X, y):
        """Optuna objective for LightGBM."""
        import lightgbm as lgb

        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "random_state": RANDOM_SEED,
            "verbose": -1,
            "n_jobs": -1,
        }

        kfold = KFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = []

        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )

            y_pred = np.clip(model.predict(X_val), 0, None)
            scores.append(r2_score(y_val, y_pred))

        return np.mean(scores)

    def _xgboost_objective(self, trial, X, y):
        """Optuna objective for XGBoost."""
        import xgboost as xgb

        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": RANDOM_SEED,
            "verbosity": 0,
            "n_jobs": -1,
            "early_stopping_rounds": 50,
        }

        kfold = KFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = []

        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            y_pred = np.clip(model.predict(X_val), 0, None)
            scores.append(r2_score(y_val, y_pred))

        return np.mean(scores)

    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """Run Optuna optimization for all models."""
        logger.info("=" * 60)
        logger.info("PHASE 6: HYPERPARAMETER OPTIMIZATION")
        logger.info("=" * 60)

        objectives = {
            "CatBoost": self._catboost_objective,
            "LightGBM": self._lightgbm_objective,
            "XGBoost": self._xgboost_objective,
        }

        for model_name, objective_fn in objectives.items():
            try:
                logger.info(f"\n  Optimizing {model_name} ({self.n_trials} trials)...")

                study = optuna.create_study(
                    direction="maximize",
                    sampler=TPESampler(seed=RANDOM_SEED),
                    study_name=f"{model_name}_optimization"
                )

                study.optimize(
                    lambda trial: objective_fn(trial, X, y),
                    n_trials=self.n_trials,
                    show_progress_bar=True,
                )

                self.studies[model_name] = study
                self.best_params[model_name] = study.best_params
                best_r2 = study.best_value

                logger.info(f"  {model_name} best R²: {best_r2:.4f}")
                logger.info(f"  {model_name} best params: {study.best_params}")

            except ImportError:
                logger.warning(f"  {model_name} not installed, skipping.")
            except Exception as e:
                logger.error(f"  {model_name} optimization failed: {e}")

        # Save best parameters
        params_path = MODELS_DIR / "best_params.json"
        with open(params_path, "w") as f:
            json.dump(self.best_params, f, indent=2, default=str)
        logger.info(f"\n  Best parameters saved to {params_path}")

        logger.info("Phase 6 complete.")
        return self.best_params

    def train_optimized_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train final models using optimized hyperparameters."""
        logger.info("\n  Training models with optimized parameters...")

        if "CatBoost" in self.best_params:
            from catboost import CatBoostRegressor
            params = self.best_params["CatBoost"].copy()
            params["random_seed"] = RANDOM_SEED
            params["verbose"] = 0
            model = CatBoostRegressor(**params)
            model.fit(X, y)
            self.best_models["CatBoost"] = model
            joblib.dump(model, MODELS_DIR / "catboost_tuned.joblib")
            logger.info("    CatBoost trained and saved.")

        if "LightGBM" in self.best_params:
            import lightgbm as lgb
            params = self.best_params["LightGBM"].copy()
            params["random_state"] = RANDOM_SEED
            params["verbose"] = -1
            params["n_jobs"] = -1
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)
            self.best_models["LightGBM"] = model
            joblib.dump(model, MODELS_DIR / "lightgbm_tuned.joblib")
            logger.info("    LightGBM trained and saved.")

        if "XGBoost" in self.best_params:
            import xgboost as xgb
            params = self.best_params["XGBoost"].copy()
            params["random_state"] = RANDOM_SEED
            params["verbosity"] = 0
            params["n_jobs"] = -1
            # Remove early_stopping_rounds for final training without eval_set
            params.pop("early_stopping_rounds", None)
            model = xgb.XGBRegressor(**params)
            model.fit(X, y)
            self.best_models["XGBoost"] = model
            joblib.dump(model, MODELS_DIR / "xgboost_tuned.joblib")
            logger.info("    XGBoost trained and saved.")

        return self.best_models
