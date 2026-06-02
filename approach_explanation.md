# Traffic Demand Prediction System — Technical Approach & Architecture

This document provides a comprehensive overview of the approach, feature engineering methodology, tools used, and relevant source files for the Traffic Demand Prediction system.

---

## 1. Project Overview & Strategy

The objective of this system is to predict traffic demand for unseen test data, maximizing the $R^2$ score under the evaluation metric:
$$\text{Score} = \max(0, 100 \times R^2(\text{actual}, \text{predicted}))$$

To achieve a production-grade, highly accurate prediction model, we designed a **10-phase machine learning pipeline** culminating in a **Stacking Ensemble model** that achieves a cross-validation $R^2$ of **0.9449**. The predictive engine is deployed via a high-performance **FastAPI backend** and visualized using a modern, responsive **Next.js dashboard**.

```
┌────────────────────────┐      ┌─────────────────────────┐      ┌──────────────────────┐
│  Raw Data Ingestion    │ ───> │   Feature Engineering   │ ───> │  Feature Selection   │
│  & Validation (Ph 1)   │      │  (30+ Features) (Ph 3)  │      │  & Rankings (Ph 4)   │
└────────────────────────┘      └─────────────────────────┘      └──────────────────────┘
                                                                            │
┌────────────────────────┐      ┌─────────────────────────┐                 ▼
│  Stacking Ensemble     │ <─── │   Optuna HPO Tuning     │ <─── ┌──────────────────────┐
│  CatBoost/XGBoost(Ph7) │      │  (XGB, Cat, LGBM) (Ph 6)│      │  Base Model Training │
└────────────────────────┘      └─────────────────────────┘      │   (5-Fold CV) (Ph 5) │
           │                                                     └──────────────────────┘
           ▼
┌────────────────────────┐      ┌─────────────────────────┐      ┌──────────────────────┐
│  SHAP Explainability   │ ───> │   Inference Pipeline    │ ───> │ FastAPI Backend &    │
│  (Global/Local) (Ph 8) │      │  (41778 rows) (Ph 9)    │      │ Next.js UI (Ph 10)   │
└────────────────────────┘      └─────────────────────────┘      └──────────────────────┘
```

---

## 2. Feature Engineering & Selection Details

Our feature engineering pipeline (`src/features/engineering.py`) transforms raw variables into **30+ highly predictive features**, grouping them into logical categories:

### A. Time & Temporal Features
Traffic demand is highly cyclical and dependent on time. We extracted:
- **Time components**: `hour` and `minute` parsed directly from the `timestamp` column.
- **Calendar components**: `weekday` (calculated via `day % 7`), `week` (`day // 7`), `month` (`((day // 30) % 12) + 1`), and `quarter`.
- **Peak Hour flags**: Categorical binary indicators for morning peak hours (7:00–9:00 AM) and evening peak hours (5:00–7:00 PM), combined into a general `is_peak_hour` flag.
- **Day status**: `is_weekend` (where `weekday >= 5`).
- **Cyclic encodings**: To help models understand that hour 23 is adjacent to hour 0, we generated sine and cosine transformations of the temporal components:
  - `hour_sin`, `hour_cos`
  - `weekday_sin`, `weekday_cos`

### B. Geospatial Features
The dataset contains a `geohash` representing location. Instead of raw strings or simple categorical encoding, we designed:
- **Geohash Decoder**: A pure Python base32 decoder that resolves any arbitrary geohash into continuous `latitude` and `longitude` coordinates.
- **Geo Frequency Encoding**: Measures how frequently a location appears in the dataset (`geo_frequency`), representing the spatial density of observations.
- **Spatial Clustering**: Performs KMeans clustering on coordinates for $K \in \{5, 10, 15\}$ to group geographic locations into spatial regions (`region_cluster_5`, `region_cluster_10`, `region_cluster_15`).

### C. Infrastructure Features
Attributes of the roadway dictate capacity and traffic patterns:
- **Road Type Encoding**: Converts categorical road type (`Residential`, `Street`, `Highway`) into ordinal integer levels (`road_type_encoded`).
- **Access flags**: Binary maps for vehicle rules (`large_vehicles_allowed`) and points of interest (`has_landmarks`).
- **Road Capacity Score**: Synthesizes lanes, road type, and truck access into a single numeric indicator:
  $$\text{Capacity} = \text{Road Type Encoded} \times \text{Number of Lanes} \times (1 + 0.5 \times \text{Large Vehicles Allowed})$$

### D. Weather Features
Atmospheric conditions heavily impact transit rates:
- **Weather Severity Score**: Map weather categories to ordinal severity indices (`Sunny`=0, `Foggy`=1, `Rainy`=2, `Snowy`=3).
- **Adverse Weather Flag**: Boolean indicator if weather is `Rainy` or `Snowy`.
- **Temperature Binning**: Discretizes temperature into 5 quantile-based bins to capture non-linear thermal impacts on travel.

### E. Cross-Domain Interaction Features
To model synergistic effects, we implemented explicit interaction features:
- `weather_x_capacity`: Multiplies weather severity score by road capacity score.
- `lanes_x_weather_severity`: Captures how weather impacts roads with different lane counts.
- `roadtype_x_weather`: Highlights how different road classes perform in severe weather.
- `lanes_x_temperature`: Captures lanes combined with heat/cold extremes.
- `landmark_density_score`: Interaction between geographic frequency and landmark presence.
- `landmark_lane_interaction`: Landmark existence interacted with lane availability.

### F. Feature Selection Pipeline
In `src/features/selection.py`, we execute a multi-method ranking engine to measure predictive power:
1. **Correlation Analysis**: Pearson and Spearman correlations with the target variable `demand`.
2. **Mutual Information**: Evaluates non-linear dependencies between inputs and the target.
3. **Tree-based Importances**: Fits Random Forest, LightGBM, and CatBoost models to extract Gini/impurity-based importances.
4. **Aggregate Rank Averaging**: Ranks features across all methods and calculates an average rank. Results are plotted and exported to `outputs/reports/feature_rankings.csv`.
*Note: Gradient boosted trees natively handle feature selection, so all engineered features were retained to preserve interactions.*

---

## 3. Modeling & Ensemble Strategy

Our modeling strategy evolved through rigorous validation:

1. **Baseline Benchmarking (5-Fold CV)**:
   We trained 5 distinct algorithms on baseline settings. Tree-based ensemble models outperformed linear models:
   - **XGBoost**: $R^2 \approx 0.9436$
   - **CatBoost**: $R^2 \approx 0.9404$
   - **LightGBM**: $R^2 \approx 0.9365$
   - **RandomForest**: $R^2 \approx 0.9333$
   - **Linear Regression**: $R^2 \approx 0.7938$

2. **Hyperparameter Optimization (Optuna)**:
   We optimized hyperparameters for the top 3 gradient boosting algorithms (XGBoost, CatBoost, LightGBM) over 100 trials using K-Fold cross-validation, tuning learning rates, max depth, subsample ratios, regularization parameters, and estimators.

3. **Ensemble Architecture**:
   To squeeze out additional performance and ensure generalization, we implemented two ensemble methods:
   - **Weighted Averaging**: A linear combination of the tuned models optimized via a grid search on cross-validation folds.
   - **Stacking Regressor**: A meta-learning model using a Linear Regression meta-estimator trained on out-of-fold predictions.
   
   **Ensemble Leaderboard:**
   - **Weighted Ensemble ($R^2 = 0.9448$)**: Weights: CatBoost (0.36), XGBoost (0.64).
   - **Stacking Ensemble ($R^2 = 0.9449$)**: Meta-weights: CatBoost (0.43), XGBoost (0.66), LightGBM (-0.08).
   
   *The Stacking Ensemble was selected as the final predictor due to its superior score and robust regularization.*

---

## 4. Tools & Technologies Used

### Machine Learning & Data Engineering
- **Python**: Core programming language.
- **Pandas / NumPy**: Data wrangling, parsing, and numeric computations.
- **Scikit-Learn**: Imputation, KMeans clustering, Mutual Information scoring, Stacking/Weighted ensembles, and evaluation metrics.
- **XGBoost / LightGBM / CatBoost**: High-performance gradient boosted decision trees.
- **Optuna**: Automated hyperparameter optimization framework.
- **SHAP**: Game-theoretic model explainability library.
- **Joblib**: Model serialization and storage.

### Web Backend & Dashboard UI
- **FastAPI / Uvicorn**: High-speed web server for serving analytics and predictions.
- **Next.js / React**: Web framework for building interactive user interfaces.
- **Tailwind CSS**: Utility-first CSS framework for modern, responsive glassmorphism styles.
- **Recharts**: Modular React chart library for demand forecast area charts and utilization indicators.
- **Docker / Docker Compose**: Containerization and multi-container execution.

---

## 5. Repository Directory & Source File Map

Below is a map of the repository's directory structure along with descriptions of the key source files.

```
Traffic/
├── api/
│   └── main.py                     # FastAPI backend serving predictions and precalculated analytics
├── dashboard/                      # Next.js web application
│   ├── src/
│   │   └── app/
│   │       ├── page.tsx            # Main dashboard component layout
│   │       └── layout.tsx          # Global HTML template and font loading
│   ├── package.json                # Dashboard dependencies (React, Recharts, Tailwind CSS)
│   ├── tailwind.config.ts          # Tailwind styling design guidelines
│   └── Dockerfile                  # Dashboard containerization
├── dataset/
│   ├── train.csv                   # Raw training data
│   ├── test.csv                    # Raw testing data
│   └── sample_submission.csv       # Reference submission file format
├── outputs/                        # Automatically generated directory for artifacts
│   ├── models/                     # Saved joblib files (models, feature engineer, ensemble weights)
│   ├── reports/                    # Data quality JSON and 16 PNG visualization plots from Phase 1-2
│   ├── shap/                       # SHAP explainability graphs (waterfalls, importances)
│   └── submissions/                # Final exported prediction file: `submission.csv` (41778 x 2)
├── src/                            # Machine Learning Pipeline modules
│   ├── __init__.py
│   ├── config.py                   # Centralized global constants, mapping tables, paths, and thresholds
│   ├── data/
│   │   ├── __init__.py
│   │   ├── eda.py                  # Generates 16 statistical target, temporal, spatial, and weather plots
│   │   └── ingestion.py            # Performs schema validation, null/duplicate checks, and data summary
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── shap_analysis.py        # Runs SHAP on the top model, generating global and local waterfalls
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py          # Extracted temporal, geohash, infrastructure, and interaction features
│   │   └── selection.py            # Ranks features using correlation, tree importances, and mutual info
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ensemble.py             # Computes optimal weighted combination and fits Stacking Regressor
│   │   ├── training.py             # Trains base models (XGB, Cat, LGB, RF, LR) with 5-fold CV
│   │   └── tuning.py               # Runs K-Fold cross-validated hyperparameter optimization using Optuna
│   └── pipeline/
│       ├── __init__.py
│       └── predict.py              # Loads models, applies transforms, runs ensemble predictions on test data
├── Dockerfile                      # ML Pipeline / FastAPI backend Docker build
├── docker-compose.yml              # Runs both dashboard (port 3000) and backend (port 8000) in unison
├── requirements.txt                # Python backend dependencies
└── run_pipeline.py                 # Pipeline Orchestrator CLI to execute phases 1-9
```

### Detailed Source File Breakdown

| Source File | Purpose | Key Details |
| :--- | :--- | :--- |
| [`src/config.py`](file:///Users/krishnajha/Traffic/src/config.py) | Global Configuration | Houses settings like `ROAD_TYPE_MAP`, `WEATHER_SEVERITY_MAP`, random seeds, feature definitions, and filesystem paths. |
| [`src/data/ingestion.py`](file:///Users/krishnajha/Traffic/src/data/ingestion.py) | Phase 1 Ingestion | Reads raw CSVs, performs column validation, checks for missing/duplicated rows, and outputs quality summaries. |
| [`src/data/eda.py`](file:///Users/krishnajha/Traffic/src/data/eda.py) | Phase 2 Visualizations | Uses Matplotlib/Seaborn to generate demand distributions, hourly profiles, geohash mappings, and weather plots. |
| [`src/features/engineering.py`](file:///Users/krishnajha/Traffic/src/features/engineering.py) | Phase 3 Features | Standardizes raw fields, decodes geohashes into latitude/longitude, groups regions via KMeans, computes road capacities, and builds interaction terms. |
| [`src/features/selection.py`](file:///Users/krishnajha/Traffic/src/features/selection.py) | Phase 4 Feature Selection | Measures linear correlation, mutual information, and forest-based importances to save rank averages. |
| [`src/models/training.py`](file:///Users/krishnajha/Traffic/src/models/training.py) | Phase 5 Model Training | Orchestrates cross-validation and trains baseline XGBoost, CatBoost, LightGBM, Random Forest, and Linear Regression. |
| [`src/models/tuning.py`](file:///Users/krishnajha/Traffic/src/models/tuning.py) | Phase 6 Tuning | Coordinates Optuna parameter sweeps. Uses cross-validation to minimize RMSE across iterations. |
| [`src/models/ensemble.py`](file:///Users/krishnajha/Traffic/src/models/ensemble.py) | Phase 7 Ensemble | Optimizes weighted models and compiles a Stacking meta-regressor, storing model parameters. |
| [`src/explainability/shap_analysis.py`](file:///Users/krishnajha/Traffic/src/explainability/shap_analysis.py) | Phase 8 SHAP Explain | Computes and plots SHAP values to illustrate features driving high and low demand. |
| [`src/pipeline/predict.py`](file:///Users/krishnajha/Traffic/src/pipeline/predict.py) | Phase 9 Prediction | Loads models, transforms the test dataset, computes final predictions, and writes output to `outputs/submissions/submission.csv`. |
| [`run_pipeline.py`](file:///Users/krishnajha/Traffic/run_pipeline.py) | Pipeline orchestrator | CLI wrapper permitting full or partial pipeline execution (e.g., `python run_pipeline.py --phase 3 4`). |
| [`api/main.py`](file:///Users/krishnajha/Traffic/api/main.py) | Backend REST Service | Exposes inference endpoints alongside precalculated statistics caching for the Next.js visualizer. |
| [`dashboard/src/app/page.tsx`](file:///Users/krishnajha/Traffic/dashboard/src/app/page.tsx) | Frontend Interface | The primary Next.js page detailing charts (forecasts, heatmap, utilization gauges, and peak periods). |
