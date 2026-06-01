"""
Phase 1: Data Ingestion Pipeline
Loads datasets, validates schema, detects anomalies, and generates quality reports.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

from src.config import (
    TRAIN_FILE, TEST_FILE, SAMPLE_SUBMISSION_FILE,
    TRAIN_COLUMNS, TEST_COLUMNS, TARGET, REPORTS_DIR
)

logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles data loading, validation, and quality reporting."""

    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.sample_submission = None
        self.quality_report = {}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all CSV datasets."""
        logger.info("Loading training data...")
        self.train_df = pd.read_csv(TRAIN_FILE)
        logger.info(f"  Train shape: {self.train_df.shape}")

        logger.info("Loading test data...")
        self.test_df = pd.read_csv(TEST_FILE)
        logger.info(f"  Test shape: {self.test_df.shape}")

        logger.info("Loading sample submission...")
        self.sample_submission = pd.read_csv(SAMPLE_SUBMISSION_FILE)
        logger.info(f"  Submission shape: {self.sample_submission.shape}")

        return self.train_df, self.test_df, self.sample_submission

    def validate_schema(self) -> Dict[str, Any]:
        """Verify expected columns exist in datasets."""
        report = {"train": {}, "test": {}}

        # Train columns
        train_cols = list(self.train_df.columns)
        expected = TRAIN_COLUMNS
        missing = [c for c in expected if c not in train_cols]
        extra = [c for c in train_cols if c not in expected]
        report["train"]["expected_columns"] = expected
        report["train"]["actual_columns"] = train_cols
        report["train"]["missing_columns"] = missing
        report["train"]["extra_columns"] = extra
        report["train"]["schema_valid"] = len(missing) == 0

        # Test columns
        test_cols = list(self.test_df.columns)
        expected_test = TEST_COLUMNS
        missing_test = [c for c in expected_test if c not in test_cols]
        extra_test = [c for c in test_cols if c not in expected_test]
        report["test"]["expected_columns"] = expected_test
        report["test"]["actual_columns"] = test_cols
        report["test"]["missing_columns"] = missing_test
        report["test"]["extra_columns"] = extra_test
        report["test"]["schema_valid"] = len(missing_test) == 0

        status = "PASS" if report["train"]["schema_valid"] and report["test"]["schema_valid"] else "FAIL"
        logger.info(f"Schema validation: {status}")

        self.quality_report["schema_validation"] = report
        return report

    def detect_missing(self) -> Dict[str, Any]:
        """Detect and report missing values."""
        report = {}

        for name, df in [("train", self.train_df), ("test", self.test_df)]:
            missing_counts = df.isnull().sum()
            missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
            cols_with_missing = missing_counts[missing_counts > 0]

            report[name] = {
                "total_rows": len(df),
                "total_missing_cells": int(df.isnull().sum().sum()),
                "columns_with_missing": {
                    col: {
                        "count": int(cols_with_missing[col]),
                        "percentage": float(missing_pct[col])
                    }
                    for col in cols_with_missing.index
                }
            }
            logger.info(f"  {name}: {len(cols_with_missing)} columns with missing values")

        self.quality_report["missing_values"] = report
        return report

    def detect_duplicates(self) -> Dict[str, Any]:
        """Check for duplicate records."""
        report = {}

        for name, df in [("train", self.train_df), ("test", self.test_df)]:
            # Full row duplicates
            n_duplicates = int(df.duplicated().sum())
            # Duplicates by key columns (excluding target for train)
            key_cols = ["geohash", "day", "timestamp"]
            n_key_duplicates = int(df.duplicated(subset=key_cols).sum())

            report[name] = {
                "total_rows": len(df),
                "full_duplicates": n_duplicates,
                "key_duplicates": n_key_duplicates,
                "key_columns_used": key_cols
            }
            logger.info(f"  {name}: {n_duplicates} full duplicates, {n_key_duplicates} key duplicates")

        self.quality_report["duplicates"] = report
        return report

    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive data summary report."""
        report = {}

        for name, df in [("train", self.train_df), ("test", self.test_df)]:
            summary = {
                "shape": {"rows": df.shape[0], "columns": df.shape[1]},
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            }

            # Numerical summary
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                num_stats = df[num_cols].describe().round(4).to_dict()
                summary["numerical_statistics"] = num_stats

            # Categorical summary
            cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
            if cat_cols:
                cat_stats = {}
                for col in cat_cols:
                    cat_stats[col] = {
                        "unique_values": int(df[col].nunique()),
                        "top_values": df[col].value_counts().head(5).to_dict(),
                        "missing": int(df[col].isnull().sum())
                    }
                summary["categorical_statistics"] = cat_stats

            report[name] = summary

        # Target-specific stats
        if TARGET in self.train_df.columns:
            target_series = self.train_df[TARGET]
            report["target_analysis"] = {
                "mean": float(target_series.mean()),
                "median": float(target_series.median()),
                "std": float(target_series.std()),
                "min": float(target_series.min()),
                "max": float(target_series.max()),
                "skewness": float(target_series.skew()),
                "kurtosis": float(target_series.kurtosis()),
                "quantiles": {
                    "25%": float(target_series.quantile(0.25)),
                    "50%": float(target_series.quantile(0.50)),
                    "75%": float(target_series.quantile(0.75)),
                    "90%": float(target_series.quantile(0.90)),
                    "95%": float(target_series.quantile(0.95)),
                    "99%": float(target_series.quantile(0.99)),
                }
            }

        self.quality_report["data_summary"] = report
        return report

    def save_reports(self):
        """Save all quality reports to files."""
        # Save full quality report as JSON
        report_path = REPORTS_DIR / "data_quality_report.json"
        with open(report_path, "w") as f:
            json.dump(self.quality_report, f, indent=2, default=str)
        logger.info(f"Data quality report saved to {report_path}")

        # Save feature summary as CSV
        feature_summary = []
        for col in self.train_df.columns:
            row = {
                "feature": col,
                "dtype": str(self.train_df[col].dtype),
                "missing_count": int(self.train_df[col].isnull().sum()),
                "missing_pct": round(self.train_df[col].isnull().sum() / len(self.train_df) * 100, 2),
                "unique_values": int(self.train_df[col].nunique()),
            }
            if self.train_df[col].dtype in [np.float64, np.int64, float, int]:
                row["mean"] = round(float(self.train_df[col].mean()), 4)
                row["std"] = round(float(self.train_df[col].std()), 4)
                row["min"] = float(self.train_df[col].min())
                row["max"] = float(self.train_df[col].max())
            feature_summary.append(row)

        summary_df = pd.DataFrame(feature_summary)
        summary_path = REPORTS_DIR / "feature_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Feature summary saved to {summary_path}")

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute the full ingestion pipeline."""
        logger.info("=" * 60)
        logger.info("PHASE 1: DATA INGESTION")
        logger.info("=" * 60)

        self.load_data()

        logger.info("\n--- Schema Validation ---")
        self.validate_schema()

        logger.info("\n--- Missing Value Detection ---")
        self.detect_missing()

        logger.info("\n--- Duplicate Detection ---")
        self.detect_duplicates()

        logger.info("\n--- Data Summary ---")
        self.generate_summary()

        self.save_reports()

        logger.info("\nPhase 1 complete.")
        return self.train_df, self.test_df
