#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 15:10:42 2025

@author: zachnewman
"""

import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.metrics import mean_squared_error
import random
from src.modeling import train_xgboost, time_series_cv_split


def tune_xgboost_with_cv(X, y, param_grid, n_splits=5, test_size=0.1,
                         max_iter=None, random_search=True, seed=42):
    results = []
    rng = random.Random(seed)

    if random_search:
        configs = list(ParameterSampler(param_grid, n_iter=max_iter or 10, random_state=seed))
    else:
        configs = list(ParameterGrid(param_grid))
        if max_iter:
            configs = configs[:max_iter]

    for i, config in enumerate(configs):
        print(f"🔁 Testing config {i+1}/{len(configs)}: {config}")
        mses = []

        for fold, (train_idx, test_idx) in enumerate(time_series_cv_split(X, y, n_splits, test_size)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            model = train_xgboost(X_train, y_train, params=config)
            y_pred = model.predict(X_test)

            mses.append(mean_squared_error(y_test, y_pred))

        results.append({
            "params": config,
            "mean_mse": np.mean(mses),
            "std_mse": np.std(mses)
        })

    results.sort(key=lambda r: r['mean_mse'], reverse=True)

    return results
