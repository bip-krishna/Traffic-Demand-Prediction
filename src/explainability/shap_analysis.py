"""
Phase 8: Model Explainability
SHAP-based feature importance and prediction explanations.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
import joblib
from typing import Dict

from src.config import SHAP_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """Generates SHAP explanations for model predictions."""

    def __init__(self):
        self.explainers = {}
        self.shap_values = {}

    def explain(self, model, X: np.ndarray, feature_names: list,
                model_name: str = "model", max_samples: int = 1000):
        """Generate SHAP explanations for a model."""
        logger.info("=" * 60)
        logger.info("PHASE 8: EXPLAINABILITY (SHAP)")
        logger.info("=" * 60)

        try:
            import shap
        except ImportError:
            logger.warning("SHAP not installed. Skipping explainability.")
            return

        logger.info(f"  Computing SHAP values for {model_name}...")

        # Use a sample for speed
        if len(X) > max_samples:
            sample_idx = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
            X_sample = X[sample_idx]
        else:
            X_sample = X

        # Create explainer
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_sample)
        except Exception as e:
            logger.warning(f"  TreeExplainer failed: {e}. Trying KernelExplainer...")
            try:
                bg = shap.sample(X_sample, min(100, len(X_sample)))
                explainer = shap.KernelExplainer(model.predict, bg)
                sv = explainer.shap_values(X_sample[:200])
                X_sample = X_sample[:200]
            except Exception as e2:
                logger.error(f"  SHAP computation failed: {e2}")
                return

        self.shap_values[model_name] = sv

        # ─── Global Feature Importance (Top 20) ─────────────────────────────
        logger.info("  Generating global feature importance plot...")

        mean_abs_shap = np.abs(sv).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-20:][::-1]

        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = [feature_names[i] for i in top_indices]
        top_values = mean_abs_shap[top_indices]

        ax.barh(range(len(top_features)), top_values[::-1],
                color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features))))
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features[::-1])
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title(f"Top 20 Feature Importance — {model_name}")
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"shap_global_importance_{model_name.lower()}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        # ─── SHAP Summary Plot (Beeswarm) ───────────────────────────────────
        logger.info("  Generating SHAP summary plot...")

        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(sv, X_sample, feature_names=feature_names,
                              show=False, max_display=20)
            plt.title(f"SHAP Summary — {model_name}")
            plt.tight_layout()
            plt.savefig(SHAP_DIR / f"shap_summary_{model_name.lower()}.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning(f"  Summary plot failed: {e}")

        # ─── Waterfall Plots for Individual Predictions ──────────────────────
        logger.info("  Generating waterfall plots for sample predictions...")

        for idx in range(min(3, len(X_sample))):
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                explanation = shap.Explanation(
                    values=sv[idx],
                    base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                    data=X_sample[idx],
                    feature_names=feature_names
                )
                shap.waterfall_plot(explanation, show=False, max_display=15)
                plt.title(f"SHAP Waterfall — Sample {idx+1}")
                plt.tight_layout()
                plt.savefig(SHAP_DIR / f"shap_waterfall_{model_name.lower()}_sample{idx+1}.png",
                            dpi=150, bbox_inches="tight")
                plt.close()
            except Exception as e:
                logger.warning(f"  Waterfall plot {idx+1} failed: {e}")

        logger.info(f"  SHAP plots saved to {SHAP_DIR}")
        logger.info("Phase 8 complete.")
