# src/tuning.py

import json
from pathlib import Path
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def tune_model(estimator, X, y, tuning_cfg):
    """
    Run hyperparameter tuning using the specified method and parameter grid.
    Returns the best parameters dict and path to saved results JSON.
    """
    method = tuning_cfg.get("method", "grid")
    param_grid = tuning_cfg.get("param_grid", {})
    cv = tuning_cfg.get("cv", 5)
    scoring = tuning_cfg.get("scoring", None)
    n_iter = tuning_cfg.get("n_iter", 10)
    random_state = tuning_cfg.get("random_state", 42)
    n_jobs = tuning_cfg.get("n_jobs", -1)
    verbose = tuning_cfg.get("verbose", 0)
    results_dir = tuning_cfg.get("results_dir", "results")
    results_filename = tuning_cfg.get("results_filename", "tuning_results.json")

    # Choose search strategy
    if method == "grid":
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )
    elif method == "random":
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown tuning method '{method}'. Use 'grid' or 'random'.")

    # Run search
    search.fit(X, y)
    best_params = search.best_params_
    cv_results = search.cv_results_

    # Save results
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_path = Path(results_dir) / results_filename
    with open(results_path, "w") as f:
        json.dump({
            "best_params": best_params,
            "cv_results": cv_results
        }, f, default=lambda obj: obj.tolist() if hasattr(obj, "tolist") else obj,
        indent=2)

    return best_params, str(results_path)