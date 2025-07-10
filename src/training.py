# src/training.py

import json
from pathlib import Path
import xgboost as xgb
import shap
import numpy as np
from .tuning import tune_model


def run_tuning(X, y, tuning_cfg, model_cfg):
    """
    Build base estimator from model_cfg and run hyperparameter tuning.
    Returns best_params dict and path to tuning results JSON.
    """
    # Construct estimator
    model_type = model_cfg.get("type", "xgboost")
    hyperparams = model_cfg.get("hyperparameters", {})
    if model_type.lower() == "xgboost":
        estimator = xgb.XGBRegressor(**hyperparams)
    else:
        raise ValueError(f"Unsupported model type '{model_type}'")

    # Delegate to tune_model
    best_params, results_path = tune_model(estimator, X, y, tuning_cfg)
    return best_params, results_path


def run_training_pipeline(X, y, tuning_results_path, training_cfg, model_cfg, selection_cfg):
    """
    Retrain the model on the full dataset using tuning results,
    compute SHAP values for feature selection, and save final model.
    Returns (estimator, shap_values, top_features).
    """
    # Load best params
    with open(tuning_results_path) as f:
        tuning_output = json.load(f)
    best_params = tuning_output.get("best_params", {})

    # Initialize and train model
    model_type = model_cfg.get("type", "xgboost")
    if model_type.lower() == "xgboost":
        estimator = xgb.XGBRegressor(**best_params)
    else:
        raise ValueError(f"Unsupported model type '{model_type}'")

    # Train with optional fit params
    fit_params = training_cfg.get("fit_params", {})
    estimator.fit(X, y, **fit_params)

    # Compute SHAP values if requested
    top_features = []
    shap_values = None
    if selection_cfg.get("method", "") == "shap":
        explainer = shap.Explainer(estimator)
        shap_values = explainer(X)
        # Mean absolute shap per feature
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        feature_names = X.columns if hasattr(X, "columns") else [f"f{i}" for i in range(len(mean_abs))]
        k = selection_cfg.get("top_k", 10)
        idxs = np.argsort(mean_abs)[-k:]
        top_features = [feature_names[i] for i in idxs]

    # Save final model
    output_dir = model_cfg.get("output_dir", "models")
    filename = model_cfg.get("filename", "final_model.json")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if hasattr(estimator, "save_model"):
        estimator.save_model(Path(output_dir) / filename)
    else:
        # fallback: save params
        with open(Path(output_dir) / filename, "w") as f:
            json.dump(estimator.get_params(), f, indent=2)

    return estimator, shap_values, top_features
