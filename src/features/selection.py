"""
Phase 4: Feature Selection
Implements correlation analysis, mutual information, and tree-based importance.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict

from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

from src.config import REPORTS_DIR, RANDOM_SEED, TARGET

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Selects the most important features using multiple methods."""

    def __init__(self):
        self.importance_rankings = {}
        self.selected_features = []

    def correlation_analysis(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Compute Pearson and Spearman correlations with target."""
        logger.info("  Running correlation analysis...")

        features_df = df[feature_cols + [TARGET]].dropna()

        # Pearson correlation with target
        pearson_corr = features_df[feature_cols].corrwith(features_df[TARGET], method="pearson")
        pearson_corr = pearson_corr.abs().sort_values(ascending=False)

        # Spearman correlation with target
        spearman_corr = features_df[feature_cols].corrwith(features_df[TARGET], method="spearman")
        spearman_corr = spearman_corr.abs().sort_values(ascending=False)

        self.importance_rankings["pearson"] = pearson_corr
        self.importance_rankings["spearman"] = spearman_corr

        # Plot correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        corr_df = pd.DataFrame({
            "Pearson": pearson_corr.head(25),
            "Spearman": spearman_corr.reindex(pearson_corr.head(25).index)
        })
        corr_df.plot(kind="barh", ax=ax, width=0.8)
        ax.set_title("Top 25 Features: Correlation with Demand")
        ax.set_xlabel("|Correlation|")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "feature_correlation.png", dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"  Top 5 Pearson: {list(pearson_corr.head(5).index)}")
        return {"pearson": pearson_corr, "spearman": spearman_corr}

    def mutual_information(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
        """Calculate mutual information scores."""
        logger.info("  Computing mutual information...")

        X = df[feature_cols].fillna(0).values
        y = df[TARGET].values

        mi_scores = mutual_info_regression(X, y, random_state=RANDOM_SEED, n_neighbors=5)
        mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)

        self.importance_rankings["mutual_info"] = mi_series

        logger.info(f"  Top 5 MI: {list(mi_series.head(5).index)}")
        return mi_series

    def tree_based_importance(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Get feature importance from tree-based models."""
        logger.info("  Computing tree-based feature importance...")

        X = df[feature_cols].fillna(0).values
        y = df[TARGET].values

        # Random Forest
        logger.info("    RandomForest importance...")
        rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                                   random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(X, y)
        rf_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
        self.importance_rankings["random_forest"] = rf_imp

        # LightGBM importance
        try:
            import lightgbm as lgb
            logger.info("    LightGBM importance...")
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                random_state=RANDOM_SEED, verbose=-1, n_jobs=-1
            )
            lgb_model.fit(X, y)
            lgb_imp = pd.Series(lgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
            # Normalize
            lgb_imp = lgb_imp / lgb_imp.sum()
            self.importance_rankings["lightgbm"] = lgb_imp
        except ImportError:
            logger.warning("    LightGBM not installed, skipping.")

        # CatBoost importance
        try:
            from catboost import CatBoostRegressor
            logger.info("    CatBoost importance...")
            cb_model = CatBoostRegressor(
                iterations=200, depth=8, learning_rate=0.1,
                random_state=RANDOM_SEED, verbose=0
            )
            cb_model.fit(X, y)
            cb_imp = pd.Series(cb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
            cb_imp = cb_imp / cb_imp.sum()
            self.importance_rankings["catboost"] = cb_imp
        except ImportError:
            logger.warning("    CatBoost not installed, skipping.")

        return self.importance_rankings

    def aggregate_rankings(self, feature_cols: List[str], top_n: int = None) -> List[str]:
        """Combine rankings from all methods using rank averaging."""
        logger.info("  Aggregating feature rankings...")

        rank_df = pd.DataFrame(index=feature_cols)

        for method, scores in self.importance_rankings.items():
            # Convert to ranks (higher score = lower rank number = better)
            rank_df[f"{method}_rank"] = scores.reindex(feature_cols).rank(ascending=False)

        # Average rank across methods
        rank_df["avg_rank"] = rank_df.mean(axis=1)
        rank_df = rank_df.sort_values("avg_rank")

        # Save rankings
        rank_df.to_csv(REPORTS_DIR / "feature_rankings.csv")

        if top_n is None:
            # Keep all features (let model handle selection)
            self.selected_features = feature_cols
        else:
            self.selected_features = rank_df.head(top_n).index.tolist()

        logger.info(f"  Total features: {len(feature_cols)}, Selected: {len(self.selected_features)}")

        # Plot top features
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = rank_df.head(30)
        ax.barh(range(len(top_features)), top_features["avg_rank"].values,
                color=sns.color_palette("viridis", len(top_features)))
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.index)
        ax.set_xlabel("Average Rank (lower is better)")
        ax.set_title("Top 30 Features by Average Rank")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "feature_rankings.png", dpi=150, bbox_inches="tight")
        plt.close()

        return self.selected_features

    def run(self, df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
        """Execute the full feature selection pipeline."""
        logger.info("=" * 60)
        logger.info("PHASE 4: FEATURE SELECTION")
        logger.info("=" * 60)

        self.correlation_analysis(df, feature_cols)
        self.mutual_information(df, feature_cols)
        self.tree_based_importance(df, feature_cols)
        selected = self.aggregate_rankings(feature_cols)

        logger.info("Phase 4 complete.")
        return selected
