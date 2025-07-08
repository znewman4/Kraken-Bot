# src/training.py

import json
from datetime import datetime
from pathlib import Path

import shap
from src.tuning                     import tune_xgboost_with_cv
from src.modeling                   import train_xgboost, prepare_features_and_target
from src.prioritise_features        import get_top_shap_features

def run_tuning(X, y):
    """
    Hyperparameter tuning via time-series CV.
    Returns (results_list, results_path).
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
    print("Best full-model MSE:", results[0]['mean_mse'])

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"xgb_tuning_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📁 Tuning results saved to {path}")
    return results, path

def run_training_pipeline(X, y, results_path):
    """
    Train final model on full data, then on top SHAP features.
    Returns (model_full, shap_values, model_top, top_features).
    """
    # 1) load best params
    with open(results_path) as f:
        results = json.load(f)
    best_params = results[0]["params"]

    # 2) train on full dataset
    model_full = train_xgboost(X, y, params=best_params)
    print("✅ Final full-data model trained")

    # 3) SHAP explanation & top features
    explainer = shap.Explainer(model_full)
    shap_values = explainer(X)
    top_features, _ = get_top_shap_features(shap_values, X.columns, top_n=10)

    # 4) train on top features
    X_top = X[top_features]
    model_top = train_xgboost(X_top, y, best_params)
    print("✅ Top-features model trained")

    return model_full, shap_values, model_top, top_features

def train_horizon_models(df, best_params, horizons=(1,5,10)):
    """
    Train and save XGBRegressor for multiple look-ahead horizons.
    """
    for h in horizons:
        print(f"🔨 Training horizon = {h}")
        X_h, y_h = prepare_features_and_target(df, horizon=h)
        m = train_xgboost(X_h, y_h, best_params)
        out = Path("models") / f"xgb_h{h}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        m.save_model(str(out))
        print(f"✅ Saved horizon-{h} model to {out}")
