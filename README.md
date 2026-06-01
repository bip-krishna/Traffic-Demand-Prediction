# 🚦 AI Traffic Demand Prediction System

A production-grade machine learning system for predicting traffic demand, featuring advanced ensemble models, SHAP explainability, and a real-time intelligence dashboard.

## 🏗 Architecture

```
Traffic/
├── src/                          # ML Pipeline
│   ├── config.py                 # Global configuration
│   ├── data/
│   │   ├── ingestion.py          # Phase 1: Data loading & validation
│   │   └── eda.py                # Phase 2: EDA visualizations
│   ├── features/
│   │   ├── engineering.py        # Phase 3: Feature engineering
│   │   └── selection.py          # Phase 4: Feature selection
│   ├── models/
│   │   ├── training.py           # Phase 5: Model development
│   │   ├── tuning.py             # Phase 6: Optuna optimization
│   │   └── ensemble.py           # Phase 7: Ensemble learning
│   ├── explainability/
│   │   └── shap_analysis.py      # Phase 8: SHAP analysis
│   └── pipeline/
│       └── predict.py            # Phase 9: Inference pipeline
├── api/                          # FastAPI Backend
│   └── main.py                   # REST API endpoints
├── dashboard/                    # Phase 10: Next.js Dashboard
├── dataset/                      # CSV datasets
├── outputs/                      # Generated outputs
│   ├── reports/                  # EDA plots & reports
│   ├── models/                   # Saved models
│   ├── submissions/              # Prediction CSVs
│   └── shap/                     # SHAP plots
├── run_pipeline.py               # Main orchestrator
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker image
└── docker-compose.yml            # Multi-service deployment
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
# Run all 9 phases (includes Optuna tuning with 100 trials)
python run_pipeline.py

# Skip Optuna tuning for faster iteration
python run_pipeline.py --skip-tuning

# Run specific phases only
python run_pipeline.py --phase 1 2 3

# Custom number of Optuna trials
python run_pipeline.py --trials 50
```

### 3. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Start the Dashboard

```bash
cd dashboard
npm install
npm run dev
```

### 5. Docker Deployment

```bash
docker-compose up --build
```

## 📊 ML Pipeline Phases

| Phase | Description | Output |
|-------|-------------|--------|
| 1 | Data Ingestion & Validation | Quality reports |
| 2 | Exploratory Data Analysis | 16+ visualizations |
| 3 | Feature Engineering | 30+ engineered features |
| 4 | Feature Selection | Importance rankings |
| 5 | Model Development | 5-fold CV results |
| 6 | Hyperparameter Optimization | Optuna-tuned models |
| 7 | Ensemble Learning | Weighted/Stacking ensemble |
| 8 | Explainability | SHAP plots |
| 9 | Prediction Pipeline | submission.csv |

## 🤖 Models

- **Baseline**: Linear Regression, Random Forest
- **Advanced**: CatBoost, LightGBM, XGBoost
- **Ensemble**: Weighted + Stacking with Ridge meta-learner
- **Optimization**: 100 Optuna trials per model

## 📈 Dashboard Features

- **Demand Forecasting**: 24-hour demand predictions with confidence intervals
- **Traffic Heatmap**: Geographic demand visualization
- **Peak Hour Detection**: Congestion period identification
- **Weather Impact**: Traffic behavior under different conditions
- **Road Capacity**: Infrastructure utilization analysis

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/predict` | Single prediction |
| GET | `/api/forecast` | 24-hour forecast |
| GET | `/api/analytics/demand-by-hour` | Hourly patterns |
| GET | `/api/analytics/demand-by-weather` | Weather impact |
| GET | `/api/analytics/heatmap` | Geographic data |
| GET | `/api/analytics/road-capacity` | Road utilization |
| GET | `/api/analytics/peak-hours` | Peak detection |

## 📋 Feature Engineering

- **Temporal**: hour, minute, weekday, week, month, quarter, peak flags
- **Cyclic**: hour_sin/cos, weekday_sin/cos
- **Geospatial**: latitude, longitude, geo_frequency, KMeans clusters
- **Infrastructure**: road_capacity_score, interaction features
- **Weather**: severity_score, temperature_bin, adverse_weather
- **Landmarks**: density_score, lane_interaction

## 📝 License

MIT
