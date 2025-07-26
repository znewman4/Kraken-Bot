import itertools
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from src.modeling import train_xgboost
from src.pnl_scorer import PnLBacktestScorer

def run_custom_tuning(X, y, base_cfg):
    param_grid = base_cfg["tuning"]["param_grid"]
    logic_params = {
        "threshold_mult": [0.05, 0.1, 0.2],
        "stop_loss_atr_mult": [1.0, 1.5, 2.0],
        "take_profit_atr_mult": [1.0, 2.0, 3.0]
    }

    keys = list(param_grid.keys()) + list(logic_params.keys())
    values = list(param_grid.values()) + list(logic_params.values())
    combinations = list(itertools.product(*values))

    cv = TimeSeriesSplit(n_splits=base_cfg["tuning"]["cv"])
    scorer = PnLBacktestScorer(base_cfg)

    best_score = -np.inf
    best_config = None

    for combo in combinations:
        model_params = dict(zip(param_grid.keys(), combo[:len(param_grid)]))
        logic_override = dict(zip(logic_params.keys(), combo[len(param_grid):]))

        fold_scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model = train_xgboost(X_train, y_train, {"params": model_params})
            score = scorer(model, X_val, y_val, logic_override)
            fold_scores.append(score)

        avg_score = np.mean(fold_scores)
        print(f"Params: {combo}, Avg PnL: {avg_score:.2f}")
        if avg_score > best_score:
            best_score = avg_score
            best_config = combo

    print("Best Config:", best_config)
    print("Best PnL:", best_score)
