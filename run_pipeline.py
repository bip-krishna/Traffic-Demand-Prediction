#!/usr/bin/env python3
"""
Traffic Demand Prediction System — Main Pipeline Orchestrator
Runs all phases sequentially or individually.

Usage:
    python run_pipeline.py                   # Run all phases
    python run_pipeline.py --phase 1 2 3     # Run specific phases
    python run_pipeline.py --skip-tuning     # Skip Optuna (use defaults)
"""

import argparse
import logging
import sys
import time
import numpy as np
import joblib
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import MODELS_DIR, TARGET


def setup_logging():
    """Configure logging with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("outputs/pipeline.log", mode="w"),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Traffic Demand Prediction Pipeline")
    parser.add_argument("--phase", nargs="+", type=int, default=None,
                        help="Specific phases to run (1-9)")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="Skip Optuna hyperparameter tuning")
    parser.add_argument("--trials", type=int, default=100,
                        help="Number of Optuna trials per model")
    args = parser.parse_args()

    # Create output dirs
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    Path("outputs/submissions").mkdir(parents=True, exist_ok=True)
    Path("outputs/shap").mkdir(parents=True, exist_ok=True)

    setup_logging()
    logger = logging.getLogger(__name__)

    phases_to_run = set(args.phase) if args.phase else set(range(1, 10))
    start_time = time.time()

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║     TRAFFIC DEMAND PREDICTION SYSTEM                    ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"Phases to run: {sorted(phases_to_run)}")

    # Shared state
    train_df = None
    test_df = None
    feature_engineer = None
    feature_cols = None
    X_train = None
    y_train = None

    # ─── Phase 1: Data Ingestion ─────────────────────────────────────────
    if 1 in phases_to_run:
        from src.data.ingestion import DataIngestion
        ingestion = DataIngestion()
        train_df, test_df = ingestion.run()

    # ─── Phase 2: EDA ────────────────────────────────────────────────────
    if 2 in phases_to_run:
        if train_df is None:
            import pandas as pd
            from src.config import TRAIN_FILE
            train_df = pd.read_csv(TRAIN_FILE)

        from src.data.eda import EDAAnalyzer
        eda = EDAAnalyzer(train_df)
        eda.run()

    # ─── Phase 3: Feature Engineering ────────────────────────────────────
    if 3 in phases_to_run:
        if train_df is None:
            import pandas as pd
            from src.config import TRAIN_FILE
            train_df = pd.read_csv(TRAIN_FILE)

        from src.features.engineering import FeatureEngineer
        feature_engineer = FeatureEngineer()
        train_df = feature_engineer.fit_transform(train_df)
        feature_cols = feature_engineer.feature_columns

        logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")

        # Prepare training arrays
        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df[TARGET].values

        # Save feature engineer for later use
        joblib.dump(feature_engineer, MODELS_DIR / "feature_engineer.joblib")
        joblib.dump(feature_cols, MODELS_DIR / "feature_columns.joblib")

    # ─── Phase 4: Feature Selection ──────────────────────────────────────
    if 4 in phases_to_run:
        if X_train is None:
            # Load from saved state
            feature_engineer = joblib.load(MODELS_DIR / "feature_engineer.joblib")
            feature_cols = joblib.load(MODELS_DIR / "feature_columns.joblib")
            import pandas as pd
            from src.config import TRAIN_FILE
            train_df = pd.read_csv(TRAIN_FILE)
            train_df = feature_engineer.fit_transform(train_df)
            X_train = train_df[feature_cols].fillna(0).values
            y_train = train_df[TARGET].values

        from src.features.selection import FeatureSelector
        selector = FeatureSelector()
        selected_features = selector.run(train_df, feature_cols)
        # We keep all features (let gradient boosting handle it)
        logger.info(f"Features retained: {len(selected_features)}")

    # ─── Phase 5: Model Development ──────────────────────────────────────
    if 5 in phases_to_run:
        if X_train is None:
            feature_engineer = joblib.load(MODELS_DIR / "feature_engineer.joblib")
            feature_cols = joblib.load(MODELS_DIR / "feature_columns.joblib")
            import pandas as pd
            from src.config import TRAIN_FILE
            train_df = pd.read_csv(TRAIN_FILE)
            train_df = feature_engineer.fit_transform(train_df)
            X_train = train_df[feature_cols].fillna(0).values
            y_train = train_df[TARGET].values

        from src.models.training import ModelTrainer
        trainer = ModelTrainer()
        cv_results = trainer.cross_validate(X_train, y_train, feature_cols)
        logger.info(f"\nCV Results:\n{cv_results.to_string(index=False)}")

    # ─── Phase 6: Hyperparameter Optimization ────────────────────────────
    if 6 in phases_to_run and not args.skip_tuning:
        if X_train is None:
            feature_engineer = joblib.load(MODELS_DIR / "feature_engineer.joblib")
            feature_cols = joblib.load(MODELS_DIR / "feature_columns.joblib")
            import pandas as pd
            from src.config import TRAIN_FILE
            train_df = pd.read_csv(TRAIN_FILE)
            train_df = feature_engineer.fit_transform(train_df)
            X_train = train_df[feature_cols].fillna(0).values
            y_train = train_df[TARGET].values

        from src.models.tuning import HyperparameterTuner
        tuner = HyperparameterTuner(n_trials=args.trials)
        best_params = tuner.optimize(X_train, y_train)
        tuned_models = tuner.train_optimized_models(X_train, y_train)
    elif 6 in phases_to_run and args.skip_tuning:
        logger.info("Skipping Optuna tuning (--skip-tuning flag).")
        logger.info("Training with default parameters instead...")

        if X_train is None:
            feature_engineer = joblib.load(MODELS_DIR / "feature_engineer.joblib")
            feature_cols = joblib.load(MODELS_DIR / "feature_columns.joblib")
            import pandas as pd
            from src.config import TRAIN_FILE
            train_df = pd.read_csv(TRAIN_FILE)
            train_df = feature_engineer.fit_transform(train_df)
            X_train = train_df[feature_cols].fillna(0).values
            y_train = train_df[TARGET].values

        from src.models.training import ModelTrainer
        trainer = ModelTrainer()
        trainer.cross_validate(X_train, y_train, feature_cols)
        trainer.train_final_models(X_train, y_train, feature_cols)

    # ─── Phase 7: Ensemble Learning ──────────────────────────────────────
    if 7 in phases_to_run:
        if X_train is None:
            feature_engineer = joblib.load(MODELS_DIR / "feature_engineer.joblib")
            feature_cols = joblib.load(MODELS_DIR / "feature_columns.joblib")
            import pandas as pd
            from src.config import TRAIN_FILE
            train_df = pd.read_csv(TRAIN_FILE)
            train_df = feature_engineer.fit_transform(train_df)
            X_train = train_df[feature_cols].fillna(0).values
            y_train = train_df[TARGET].values

        from src.models.ensemble import EnsembleBuilder
        ensemble = EnsembleBuilder()

        # Load tuned models
        tuned_models = {}
        for name, fname in [("CatBoost", "catboost_tuned.joblib"),
                             ("LightGBM", "lightgbm_tuned.joblib"),
                             ("XGBoost", "xgboost_tuned.joblib")]:
            path = MODELS_DIR / fname
            if path.exists():
                tuned_models[name] = joblib.load(path)

        if len(tuned_models) >= 2:
            ensemble.set_models(tuned_models)
            ensemble.select_best_ensemble(X_train, y_train)
        else:
            logger.warning("Not enough tuned models for ensemble. Need at least 2.")

    # ─── Phase 8: Explainability ─────────────────────────────────────────
    if 8 in phases_to_run:
        if X_train is None:
            feature_engineer = joblib.load(MODELS_DIR / "feature_engineer.joblib")
            feature_cols = joblib.load(MODELS_DIR / "feature_columns.joblib")
            import pandas as pd
            from src.config import TRAIN_FILE
            train_df = pd.read_csv(TRAIN_FILE)
            train_df = feature_engineer.fit_transform(train_df)
            X_train = train_df[feature_cols].fillna(0).values
            y_train = train_df[TARGET].values

        from src.explainability.shap_analysis import SHAPExplainer

        # Use the best single model for SHAP (typically CatBoost or LightGBM)
        best_model = None
        best_name = None
        for name in ["CatBoost", "LightGBM", "XGBoost"]:
            path = MODELS_DIR / f"{name.lower()}_tuned.joblib"
            if path.exists():
                best_model = joblib.load(path)
                best_name = name
                break

        if best_model is not None:
            explainer = SHAPExplainer()
            explainer.explain(best_model, X_train, feature_cols, best_name)
        else:
            logger.warning("No tuned model found for SHAP analysis.")

    # ─── Phase 9: Prediction ─────────────────────────────────────────────
    if 9 in phases_to_run:
        if feature_engineer is None:
            feature_engineer = joblib.load(MODELS_DIR / "feature_engineer.joblib")
        if feature_cols is None:
            feature_cols = joblib.load(MODELS_DIR / "feature_columns.joblib")

        from src.pipeline.predict import PredictionPipeline
        pipeline = PredictionPipeline()
        submission_path = pipeline.run(feature_engineer, feature_cols)
        logger.info(f"\nSubmission file: {submission_path}")

    # ─── Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info(f"PIPELINE COMPLETE — Total time: {elapsed/60:.1f} minutes")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
