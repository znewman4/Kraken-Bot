import numpy as np
import xgboost as xgb


def prepare_features_and_target(df, model_cfg):
    """
    Build feature matrix X and regression target y.
    y_t = (close_{t+horizon} - close_t)/close_t

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
    """
    horizon = model_cfg.get("horizon", 1)
    df2 = df.copy()
    df2['target'] = (df2['close'].shift(-horizon) - df2['close']) / df2['close']
    df2.dropna(inplace=True)
    feature_cols = [c for c in df2.columns if c not in ('target', 'close')]
    X = df2[feature_cols]
    y = df2['target']
    return X, y


def time_series_cv_split(n_samples, train_size, test_size, step=1):
    """
    Generate expanding-window time-series cross-validation indices.

    Args:
        n_samples (int): Total number of samples in the dataset.
        train_size (int): Initial training window size.
        test_size (int): Size of each test window.
        step (int): Number of folds to generate.

    Yields:
        train_idx (np.ndarray), test_idx (np.ndarray)
    """
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
    """
    Trains an XGBoost regressor on the training set with hyperparameters
    defined in model_cfg. Overrides default settings with model_cfg['params'].

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        model_cfg (dict): Model configuration dict, may include keys:
            - n_estimators, learning_rate, max_depth, subsample,
            - colsample_bytree, gamma, reg_lambda,
            - params (dict) for additional XGB parameters.

    Returns:
        model (xgb.XGBRegressor): Fitted XGBoost model.
    """
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
