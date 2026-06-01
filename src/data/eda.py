"""
Phase 2: Exploratory Data Analysis
Generates comprehensive visualizations and statistical insights.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

from src.config import REPORTS_DIR, TARGET

logger = logging.getLogger(__name__)

# Style configuration
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


class EDAAnalyzer:
    """Generates EDA visualizations and statistical reports."""

    def __init__(self, train_df: pd.DataFrame):
        self.df = train_df.copy()
        self.output_dir = REPORTS_DIR
        # Pre-parse timestamp for time analysis
        self._parse_timestamp()

    def _parse_timestamp(self):
        """Parse timestamp column into hour and minute."""
        if "timestamp" in self.df.columns:
            parts = self.df["timestamp"].astype(str).str.split(":", expand=True)
            self.df["_hour"] = parts[0].astype(int)
            self.df["_minute"] = parts[1].astype(int) if parts.shape[1] > 1 else 0

    def _save_plot(self, name: str):
        """Save current plot to output directory."""
        path = self.output_dir / f"{name}.png"
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info(f"  Saved: {path.name}")

    # ─── Target Analysis ─────────────────────────────────────────────────────

    def plot_demand_distribution(self):
        """Demand distribution histogram with KDE."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram + KDE
        sns.histplot(self.df[TARGET], bins=80, kde=True, ax=axes[0],
                     color="#4C72B0", edgecolor="white", alpha=0.7)
        axes[0].set_title("Demand Distribution")
        axes[0].set_xlabel("Demand")
        axes[0].axvline(self.df[TARGET].mean(), color="red", linestyle="--",
                        label=f"Mean: {self.df[TARGET].mean():.4f}")
        axes[0].axvline(self.df[TARGET].median(), color="orange", linestyle="--",
                        label=f"Median: {self.df[TARGET].median():.4f}")
        axes[0].legend()

        # Log-transformed distribution
        log_demand = np.log1p(self.df[TARGET])
        sns.histplot(log_demand, bins=80, kde=True, ax=axes[1],
                     color="#55A868", edgecolor="white", alpha=0.7)
        axes[1].set_title("Log(1 + Demand) Distribution")
        axes[1].set_xlabel("Log(1 + Demand)")

        self._save_plot("01_demand_distribution")

    def plot_demand_outliers(self):
        """Box plot and violin plot for demand outliers."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.boxplot(y=self.df[TARGET], ax=axes[0], color="#4C72B0", width=0.3)
        axes[0].set_title("Demand Outliers (Box Plot)")

        sns.violinplot(y=self.df[TARGET], ax=axes[1], color="#55A868", inner="quartile")
        axes[1].set_title("Demand Distribution (Violin Plot)")

        self._save_plot("02_demand_outliers")

    def plot_demand_quantiles(self):
        """Quantile analysis table and plot."""
        quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        values = [self.df[TARGET].quantile(q) for q in quantiles]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([f"{int(q*100)}%" for q in quantiles], values, color=sns.color_palette("viridis", len(quantiles)))
        ax.set_title("Demand Quantiles")
        ax.set_xlabel("Quantile")
        ax.set_ylabel("Demand Value")
        for i, (q, v) in enumerate(zip(quantiles, values)):
            ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)

        self._save_plot("03_demand_quantiles")

    # ─── Time Analysis ───────────────────────────────────────────────────────

    def plot_demand_vs_hour(self):
        """Average demand by hour of day."""
        if "_hour" not in self.df.columns:
            return

        hourly = self.df.groupby("_hour")[TARGET].agg(["mean", "std"]).reset_index()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(hourly["_hour"], hourly["mean"], marker="o", linewidth=2,
                color="#4C72B0", markersize=6)
        ax.fill_between(hourly["_hour"],
                        hourly["mean"] - hourly["std"],
                        hourly["mean"] + hourly["std"],
                        alpha=0.2, color="#4C72B0")
        ax.set_title("Average Demand by Hour of Day")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Mean Demand")
        ax.set_xticks(range(24))

        self._save_plot("04_demand_vs_hour")

    def plot_demand_vs_day(self):
        """Demand distribution across different days."""
        if "day" not in self.df.columns:
            return

        daily = self.df.groupby("day")[TARGET].mean().reset_index()

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(daily["day"], daily[TARGET], color="#55A868", alpha=0.8)
        ax.set_title("Average Demand by Day")
        ax.set_xlabel("Day")
        ax.set_ylabel("Mean Demand")

        self._save_plot("05_demand_vs_day")

    def plot_demand_vs_weekday(self):
        """Demand by weekday (derived from day column)."""
        if "day" not in self.df.columns:
            return

        self.df["_weekday"] = self.df["day"] % 7
        weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        weekday_demand = self.df.groupby("_weekday")[TARGET].mean().reset_index()
        weekday_demand["weekday_name"] = weekday_demand["_weekday"].map(
            lambda x: weekday_names[x]
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#4C72B0"] * 5 + ["#C44E52"] * 2  # weekdays blue, weekend red
        ax.bar(weekday_demand["weekday_name"], weekday_demand[TARGET], color=colors)
        ax.set_title("Average Demand by Day of Week")
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Mean Demand")

        self._save_plot("06_demand_vs_weekday")

    def plot_demand_vs_month(self):
        """Demand patterns across months (estimated from day)."""
        if "day" not in self.df.columns:
            return

        self.df["_month"] = (self.df["day"] // 30).clip(upper=11) + 1
        monthly = self.df.groupby("_month")[TARGET].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(monthly["_month"], monthly[TARGET], color=sns.color_palette("coolwarm", len(monthly)))
        ax.set_title("Average Demand by Month (Estimated)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Mean Demand")

        self._save_plot("07_demand_vs_month")

    # ─── Spatial Analysis ────────────────────────────────────────────────────

    def plot_demand_vs_geohash(self):
        """Top geohash locations by demand."""
        if "geohash" not in self.df.columns:
            return

        geo_demand = self.df.groupby("geohash")[TARGET].mean().sort_values(ascending=False).head(30)

        fig, ax = plt.subplots(figsize=(14, 6))
        geo_demand.plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_title("Top 30 Geohash Locations by Mean Demand")
        ax.set_xlabel("Mean Demand")
        ax.set_ylabel("Geohash")
        ax.invert_yaxis()

        self._save_plot("08_demand_vs_geohash")

    def plot_demand_vs_region(self):
        """Demand by geohash prefix (approximate region)."""
        if "geohash" not in self.df.columns:
            return

        self.df["_geo_prefix"] = self.df["geohash"].str[:4]
        region_demand = self.df.groupby("_geo_prefix")[TARGET].agg(["mean", "count"]).reset_index()
        region_demand = region_demand.sort_values("mean", ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(region_demand["_geo_prefix"], region_demand["mean"],
                      color=sns.color_palette("magma", len(region_demand)))
        ax.set_title("Average Demand by Region (Geohash Prefix)")
        ax.set_xlabel("Region")
        ax.set_ylabel("Mean Demand")
        plt.xticks(rotation=45)

        self._save_plot("09_demand_vs_region")

    # ─── Weather Analysis ────────────────────────────────────────────────────

    def plot_demand_vs_weather(self):
        """Demand distribution by weather condition."""
        if "Weather" not in self.df.columns:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Violin plot
        weather_order = ["Sunny", "Foggy", "Rainy", "Snowy"]
        valid = self.df[self.df["Weather"].isin(weather_order)]
        sns.violinplot(x="Weather", y=TARGET, data=valid, order=weather_order,
                       ax=axes[0], palette="Set2", inner="quartile")
        axes[0].set_title("Demand Distribution by Weather")

        # Mean demand bar chart
        weather_mean = valid.groupby("Weather")[TARGET].mean().reindex(weather_order)
        axes[1].bar(weather_mean.index, weather_mean.values,
                    color=["#FFD700", "#A9A9A9", "#4682B4", "#87CEEB"])
        axes[1].set_title("Average Demand by Weather")
        axes[1].set_ylabel("Mean Demand")

        self._save_plot("10_demand_vs_weather")

    def plot_demand_vs_temperature(self):
        """Demand vs temperature scatter with regression."""
        if "Temperature" not in self.df.columns:
            return

        valid = self.df.dropna(subset=["Temperature"])
        sample = valid.sample(min(5000, len(valid)), random_state=42)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(sample["Temperature"], sample[TARGET], alpha=0.3,
                   s=10, color="#4C72B0")

        # Add regression line
        z = np.polyfit(sample["Temperature"], sample[TARGET], 1)
        p = np.poly1d(z)
        temp_range = np.linspace(sample["Temperature"].min(), sample["Temperature"].max(), 100)
        ax.plot(temp_range, p(temp_range), color="red", linewidth=2, linestyle="--",
                label=f"Trend: y = {z[0]:.4f}x + {z[1]:.4f}")

        ax.set_title("Demand vs Temperature")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Demand")
        ax.legend()

        self._save_plot("11_demand_vs_temperature")

    # ─── Road Analysis ───────────────────────────────────────────────────────

    def plot_demand_vs_roadtype(self):
        """Demand by road type."""
        if "RoadType" not in self.df.columns:
            return

        valid = self.df.dropna(subset=["RoadType"])

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x="RoadType", y=TARGET, data=valid, ax=ax,
                    palette="Set3", order=["Residential", "Street", "Highway"])
        ax.set_title("Demand Distribution by Road Type")

        self._save_plot("12_demand_vs_roadtype")

    def plot_demand_vs_lanes(self):
        """Demand by number of lanes."""
        if "NumberofLanes" not in self.df.columns:
            return

        lane_demand = self.df.groupby("NumberofLanes")[TARGET].agg(["mean", "std"]).reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(lane_demand["NumberofLanes"].astype(str), lane_demand["mean"],
               yerr=lane_demand["std"], capsize=5,
               color=sns.color_palette("Set2", len(lane_demand)))
        ax.set_title("Average Demand by Number of Lanes")
        ax.set_xlabel("Number of Lanes")
        ax.set_ylabel("Mean Demand")

        self._save_plot("13_demand_vs_lanes")

    def plot_demand_vs_large_vehicles(self):
        """Demand by large vehicle allowance."""
        if "LargeVehicles" not in self.df.columns:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.violinplot(x="LargeVehicles", y=TARGET, data=self.df, ax=ax,
                       palette="Set2", inner="quartile")
        ax.set_title("Demand by Large Vehicle Allowance")

        self._save_plot("14_demand_vs_large_vehicles")

    def plot_demand_vs_landmarks(self):
        """Demand by landmark presence."""
        if "Landmarks" not in self.df.columns:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x="Landmarks", y=TARGET, data=self.df, ax=ax, palette="pastel")
        ax.set_title("Demand by Landmark Presence")

        self._save_plot("15_demand_vs_landmarks")

    # ─── Correlation Analysis ────────────────────────────────────────────────

    def plot_correlation_matrix(self):
        """Correlation heatmap for numerical features."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return

        corr = self.df[num_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, square=True, ax=ax, linewidths=0.5)
        ax.set_title("Feature Correlation Matrix")

        self._save_plot("16_correlation_matrix")

    # ─── Main Runner ─────────────────────────────────────────────────────────

    def run(self):
        """Execute the full EDA pipeline."""
        logger.info("=" * 60)
        logger.info("PHASE 2: EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 60)

        logger.info("\n--- Target Analysis ---")
        self.plot_demand_distribution()
        self.plot_demand_outliers()
        self.plot_demand_quantiles()

        logger.info("\n--- Time Analysis ---")
        self.plot_demand_vs_hour()
        self.plot_demand_vs_day()
        self.plot_demand_vs_weekday()
        self.plot_demand_vs_month()

        logger.info("\n--- Spatial Analysis ---")
        self.plot_demand_vs_geohash()
        self.plot_demand_vs_region()

        logger.info("\n--- Weather Analysis ---")
        self.plot_demand_vs_weather()
        self.plot_demand_vs_temperature()

        logger.info("\n--- Road Analysis ---")
        self.plot_demand_vs_roadtype()
        self.plot_demand_vs_lanes()
        self.plot_demand_vs_large_vehicles()
        self.plot_demand_vs_landmarks()

        logger.info("\n--- Correlation Analysis ---")
        self.plot_correlation_matrix()

        logger.info(f"\nAll plots saved to {self.output_dir}")
        logger.info("Phase 2 complete.")
