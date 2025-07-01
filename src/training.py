#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 15:07:02 2025

@author: zachnewman
"""

# src/training.py

import json
from datetime import datetime
from pathlib import Path
import shap
from src.tuning import tune_xgboost_with_cv
from src.modeling import train_xgboost
from src.prioritise_features import get_top_shap_features
from sklearn.metrics import mean_squared_error





def run_tuning(X, y):
    """
    Runs hyperparameter tuning on XGBoost using time series cross-validation.
    Saves results to a JSON file and returns both the results and the file path.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 4],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 1],
        'lambda': [1, 5],
        'alpha': [0, 1],
    }

    results = tune_xgboost_with_cv(X, y, param_grid)

    print("Best full model MSE:", results[0]['mean_mse'])

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"xgb_tuning_{timestamp}.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"📁 Results saved to {results_path}")

    return results, results_path

    

def run_training_pipeline(X, y, results_path):
    
    """
   Final training pipeline:
    - Loads best hyperparameters from tuning results
    - Trains XGBoost model on full dataset
    - Runs SHAP summary explanation
   """

    # Load tuning results
    with open(results_path, "r") as f:
        results = json.load(f)
    best_params = results[0]["params"]

    # Train model on full data
    model = train_xgboost(X, y, params=best_params)
    print("✅ Final model trained on full dataset.")

    # Run SHAP explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False)
    
    top_features, shap_summary = get_top_shap_features(shap_values, X.columns, top_n=10)
    X_top = X[top_features]
    
    model_top = train_xgboost(X_top, y, best_params)

    return model, shap_values, model_top, top_features



