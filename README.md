# Air Quality Index (AQI) Forecasting System

An end-to-end, serverless Machine Learning Operations (MLOps) pipeline designed to predict severe atmospheric pollution events and 3-day PM2.5 trajectories across global cities. 

**Live Dashboard:** [https://aqi-predictor-two.vercel.app/](https://aqi-predictor-two.vercel.app/)

---

## Project Overview

Traditional models predicting mean daily particulate matter (PM2.5) fail to warn users of dangerous intra-day smog spikes. A mathematically accurate "average" day can mask a highly toxic 4-hour pollution event. This system shifts the focus from simple statistical averaging to a health-first early warning system by engineering localized time-series features, mitigating hardware anomalies, and deploying a self-optimizing Model Zoo.

The entire infrastructure is fully decoupled and serverless, requiring zero manual intervention to extract data, retrain models, evaluate metrics, and update production endpoints.

---

## System Architecture

The project utilizes a completely decoupled, zero-downtime architecture divided into four main macro-components:

1. **Orchestration & ETL:** GitHub Actions executes chronologically scheduled cron jobs for daily data ingestion and model retraining.
2. **Feature Store & Model Registry:** Hopsworks acts as the central state-manager, securely housing materialized Feature Groups, offline Feature Views, and versioned model artifacts.
3. **Inference Backend:** A FastAPI application deployed on Render that manages an integrated RAM cache. It intercepts webhooks from GitHub Actions to dynamically reload newly trained models into memory without dropping incoming user requests.
4. **Stateless Frontend:** A Next.js application deployed on Vercel that handles the non-linear conversion of raw PM2.5 predictions into standard EPA Air Quality Index metrics for the end user.

---

## Data Engineering Pipeline

The feature pipeline continuously extracts 14-day rolling windows of data via the Open-Meteo API using exponential backoff protocols. 

### Forward-Fill Imputation Strategy (Synthetic Today)
To compute daily aggregations for the current, incomplete day, the pipeline queries an additional 24 hours of future weather forecast data. Because future PM2.5 data cannot be observed, the pipeline applies a strictly chronological forward-fill matrix transformation (`.ffill()`), carrying the last recorded PM2.5 measurement into the unobserved future hours. This stitches past observed pollution data with predicted meteorological patterns, allowing the pipeline to generate immediate forecasts without missing values.

### Hardware Anomaly Mitigation
The system scans for isolated, physically impossible sensor errors. For example, if PM2.5 jumps by 850 in a single hour and immediately drops, the system identifies the bi-directional spike, neutralizes the anomaly to `NaN`, and applies linear interpolation to maintain time-series continuity.

### Feature Aggregation
Cleaned hourly data is aggregated into daily means. The pipeline computes:
* **Macro-Momentum:** 3-day and 7-day rolling averages.
* **Atmospheric Velocity:** A derivative rate-of-change feature utilizing lag mechanics to measure pollution trajectory.
* **Temporal Attributes:** Extracted cyclic markers (month, day of week, day of year).

---

## Machine Learning Pipeline

The training pipeline provisions separate champion models for 1-day, 2-day, and 3-day forecasting horizons using a Global Modeling strategy (concatenating geographic data to allow the model to learn physics from diverse climates).

### The Model Zoo & Cross-Validation
The pipeline evaluates multiple algorithmic architectures:
* **Statistical:** Ridge Regression
* **Ensemble Tree Methods:** Random Forest and XGBoost
* **Deep Learning:** Multi-Layer Perceptron (MLP)

Models undergo rigorous hyperparameter tuning via `GridSearchCV` coupled with `TimeSeriesSplit` to strictly prevent temporal data leakage. 

### Implicit Feature Selection
To prevent the MLP from overfitting on uninformative features, the system utilizes L2 Regularization. By passing the `alpha` penalty parameter through the hyperparameter grid, the neural network mathematically decays the weights of noisy features toward zero, acting as an implicit feature-selector without requiring secondary PCA pipelines.

### Segmented Evaluation Metrics
Due to severe class imbalance (99.4% of days classify as "Normal"), standard RMSE metrics mask poor performance during hazardous events. The system employs a custom evaluation layer tracking Mean Absolute Error (MAE) across segmented severity buckets:

| Horizon | Condition | MAE (µg/m³) |
| :--- | :--- | :--- |
| **1-Day** | Normal (<100) | 4.77 |
| | Moderate (100-200) | 26.04 |
| **2-Day** | Normal (<100) | 6.37 |
| | Moderate (100-200) | 38.53 |
| **3-Day** | Normal (<100) | 6.82 |
| | Moderate (100-200) | 41.09 |

### Explainable AI (SHAP)
The pipeline dynamically generates SHAP values based on the winning architecture. For Deep Learning champions, the system utilizes a 50-centroid K-Means summarization of the training set before applying the `KernelExplainer`, exponentially reducing the computational overhead required to interpret the neural network.

---

## Repository Structure

```text
├── src/
│   ├── config.py
│   │
│   ├── feature_pipeline/
│   │   ├── backfill.py            # Entry point for ETL orchestration
│   │   ├── fetch_data.py          # API extraction and Synthetic Today bridge
│   │   ├── compute_features.py    # Anomaly mitigation and feature engineering
│   │   └── store_features.py      # Hopsworks Feature Group ingestion
│   │
│   ├── training_pipeline/
│       ├── fetch_training_data.py # Hopsworks Feature View materialization
│       ├── train_evaluate.py      # Model Zoo, GridSearchCV, Segmented MAE, SHAP
│       └── register_model.py      # Hopsworks Model Registry deployment
│
├── .github/workflows/
│   ├── feature_pipeline.yml       # Automated daily feature updates
│   ├── keep_alive.yml             # Daily requested to keep the backend awake
│   └── training_pipeline.yml      # Automated daily training and webhook dispatch
│
├── requirements.txt
└── README.md