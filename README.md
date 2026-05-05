# Air Quality Index (AQI) Forecasting System

This repository contains an end-to-end, serverless machine learning pipeline designed to predict the Air Quality Index (AQI) for the upcoming three days. 

## Dashboard URL
https://aqi-predictor-two.vercel.app/

## Project Overview

The goal of this project is to build an automated, low-maintenance system that processes environmental data and serves reliable forecasts. 

**Core Components:**
* **Automated Daily Data Ingestion Pipeline:** Automatically fetching live weather and pollution data from external APIs daily.
* **Feature & Model Pipelines:** Processing historical data to train and evaluate predictive models daily.
* **Automation:** Utilizing CI/CD workflows to continuously update features at every hour and retrain models every day without manual intervention.
* **Dashboard:** An interactive web interface to visualize real-time conditions, forecasts, and model insights.
