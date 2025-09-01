# modeling.py
import numpy as np
import xgboost as xgb
import pandas as pd 


def prepare_features_and_target(df, model_cfg):
    horizon = int(model_cfg.get("horizon", 1))
    df2 = df.copy()

    # simple return over horizon -> target in bps
    frac_ret = (df2['close'].shift(-horizon) - df2['close']) / df2['close']
    df2['target'] = frac_ret * 1e4  # bps

    # keep only numeric columns for features
    num = df2.select_dtypes(include=[np.number]).copy()

    # features = numeric columns except target & close
    feature_cols = [c for c in num.columns if c not in ('target', 'close')]

    # replace infs with NaN, then drop rows with any NaN in X or y
    X = num[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df2['target']
    df_xy = pd.concat([X, y], axis=1).dropna()

    # cast to float32 for consistency/perf
    X = df_xy[feature_cols].astype('float32')
    y = df_xy['target'].astype('float32')
    return X, y



def time_series_cv_split(n_samples, train_size, test_size, step=1):

    for i in range(step):
        train_end = train_size + i * test_size
        test_start = train_end
        test_end = test_start + test_size
        if test_end > n_samples:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        yield train_idx, test_idx


def train_xgboost(X_train, y_train, model_cfg):

    # Default hyperparameters
    default_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': model_cfg.get('n_estimators', 100),
        'learning_rate': model_cfg.get('learning_rate', 0.1),
        'max_depth': model_cfg.get('max_depth', 3),
        'subsample': model_cfg.get('subsample', 0.8),
        'colsample_bytree': model_cfg.get('colsample_bytree', 0.8),
        'gamma': model_cfg.get('gamma', 0),
        'reg_lambda': model_cfg.get('reg_lambda', 1),
    }
    # Override with any additional parameters
    extra_params = model_cfg.get('params', {}) or {}
    default_params.update(extra_params)

    model = xgb.XGBRegressor(**default_params)
    model.fit(X_train, y_train)
    return model
