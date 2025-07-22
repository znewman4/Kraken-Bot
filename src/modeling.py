# src/modeling.py

import numpy as np
import xgboost as xgb

def prepare_features_and_target(df, model_cfg):
    """
    Build feature matrix X and regression target y for a given horizon.
    y_t = (close_{t+horizon} - close_t) / close_t
    """
    horizon = model_cfg.get("horizon", 1)
    df2 = df.copy()
    # Shift forward so target at time t is the future return over 'horizon' bars
    df2['target'] = (df2['close'].shift(-horizon) - df2['close']) / df2['close']
    df2.dropna(inplace=True)

    # All columns except 'target' and 'close' are features
    feature_cols = [c for c in df2.columns if c not in ('target', 'close')]
    X = df2[feature_cols]
    y = df2['target']
    return X, y

def time_series_cv_split(X, y, n_splits=5, test_size=0.1):
    """
    Expanding window generator for time series CV.
    """
    n_samples = len(X)
    test_len = int(n_samples * test_size)
    train_end = n_samples - test_len * n_splits

    for i in range(n_splits):
        train_stop  = train_end + i * test_len
        test_start  = train_stop
        test_stop   = test_start + test_len
        if test_stop > n_samples:
            break
        yield np.arange(0, train_stop), np.arange(test_start, test_stop)

def train_xgboost(X_train, y_train, model_cfg):
    """
    Trains XGBoost regressor based on hyperparameters in model_cfg.
    """
    # pull hyperparams out of config (fall back on these defaults)
    default_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }

    # update with whatever user specified
    user_params = model_cfg.get("hyperparameters", {})
    default_params.update(user_params)

    model = xgb.XGBRegressor(**default_params)
    model.fit(X_train, y_train)
    return model
