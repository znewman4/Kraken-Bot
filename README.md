#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 14:15:05 2025

@author: zachnewman
"""

# Kraken Project - Bitcoin OHLCV Analysis and Prediction

## Overview

This project collects, cleans, engineers features, and models Bitcoin OHLCV (Open-High-Low-Close-Volume) data from Kraken exchange to predict price movements using XGBoost classifiers. It is structured into modular components to maintain clean, reusable, and testable code.

---

## Project Structure & Module Descriptions

### 1. `data_loading`

- **Purpose:** Fetches raw OHLCV data from Kraken and saves data to CSV files.
- **Key Functions:**
  - `fetch_ohlcv_kraken(pair, interval)`: Queries Kraken API and returns raw OHLCV data as a DataFrame.
  - `save_to_csv(df, file_path)`: Saves a DataFrame to a CSV file.
- **Used by:** `main.py` for initial data acquisition.

---

### 2. `data_cleaning`

- **Purpose:** Cleans raw OHLCV data and performs validation checks to ensure data quality.
- **Key Functions:**
  - `clean_ohlcv(df)`: Applies filtering and fixes missing or corrupted data points.
  - `validate_ohlcv(df)`: Validates dataframe structure and content consistency.
- **Used by:** `main.py` after loading raw data to prepare for feature engineering.

---

### 3. `technical_engineering`

- **Purpose:** Adds technical analysis indicators and return-based features used for machine learning.
- **Key Functions:**
  - `add_technical_indicators(df)`: Calculates indicators like RSI, moving averages, volatility, etc.
  - `add_return_features(df)`: Adds return metrics such as log returns and rolling volatility.
- **Used by:** `main.py` for feature creation after data cleaning.

---

### 4. `modeling`

- **Purpose:** Contains core machine learning utilities for preparing data, training models, and evaluating performance.
- **Key Functions:**
  - `prepare_features_and_target(df)`: Extracts feature matrix `X` and binary target vector `y`.
  - `train_xgboost(X_train, y_train, params)`: Trains an XGBoost model with specified hyperparameters.
  - `evaluate_model(X, y, n_splits, test_size, params)`: Performs time-series cross-validation, trains models, evaluates AUC scores, and plots results.
- **Used by:** 
  - `training` and `tuning` modules to build, validate, and tune models.
  - `main.py` indirectly via those modules.

---

### 5. `tuning`

- **Purpose:** Runs hyperparameter tuning using cross-validation, either full grid search or randomized search.
- **Key Functions:**
  - `tune_xgboost_with_cv(X, y, param_grid, n_splits, test_size, max_iter, random_search)`: Returns a sorted list of hyperparameter configurations ranked by mean AUC.
- **Uses:** Functions from `modeling` to train and evaluate models during tuning.
- **Used by:** `training` and `main.py` to find optimal hyperparameters.

---

### 6. `training`

- **Purpose:** Coordinates the training pipeline including tuning, final training, and model interpretation.
- **Key Functions:**
  - `run_tuning(X, y)`: Wraps the tuning process and returns the best hyperparameter results.
  - `explain_model_with_shap(X, params)`: Trains the model with best params and uses SHAP for interpretability.
  - `run_training_pipeline`: Function ran in main.py to run final model, references Shap explainer

- **Uses:** Calls `tuning` for hyperparameter search and `modeling` for training and evaluation.
- **Used by:** `main.py` to handle the training lifecycle cleanly.

---

## Execution Order in `main.py`

1. **Data Loading:**
   - Calls `fetch_ohlcv_kraken()` to get raw OHLCV data.
2. **Data Cleaning & Validation:**
   - Applies `validate_ohlcv()` then `clean_ohlcv()` and validates again.
3. **Feature Engineering:**
   - Uses `add_technical_indicators()` and `add_return_features()`.
   - Drops rows with missing data after feature engineering.
4. **Saving Engineered Data:**
   - Calls `save_to_csv()` to persist processed data.
5. **Data Preparation:**
   - Extracts features and target using `prepare_features_and_target()`.
6. **Training & Tuning:**
   - Executes `run_tuning()` from `training` module to find best hyperparameters.
7. **Explainability:**
   - pipeline will run optimal parameters and return shap_values
   


## Module Interaction Summary

- `main.py` **imports and orchestrates** the entire pipeline, calling each module in sequence.
- `training` **wraps** functionality from `tuning` and `modeling` for cleaner orchestration of the model building process.
- `tuning` **depends on** `modeling` for training and evaluation during hyperparameter search.
- `modeling` provides core ML functionality that is reused across tuning and training.
- Data-related modules (`data_loading`, `data_cleaning`, `technical_engineering`) **run sequentially** for data acquisition, quality control, and feature generation.

---

## Notes

- The dependency flow is designed to be **unidirectional**, minimizing circular imports and improving maintainability.
- This modular design allows easy extension and testing of individual components without impacting the entire codebase.

---

Feel free to reach out for clarifications or contributions!  
— Zach Newman  
