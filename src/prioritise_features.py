#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 18:16:52 2025

@author: zachnewman
"""

import pandas as pd
import numpy as np

def get_top_shap_features(shap_values, feature_names, selection_cfg):
    top_n = selection_cfg.get("top_k", 10)
    
    """
    Get top N features based on mean absolute SHAP values.

    Args:
        shap_values: SHAP values object returned from SHAP explainer (already computed)
        feature_names (list or pd.Index): Column names of the original feature matrix
        top_n (int): Number of top features to return

    Returns:
        top_features (list): Top N feature names
        shap_df (pd.DataFrame): SHAP importance summary table
    """
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False)

    top_features = shap_df["feature"].head(top_n).tolist()
    return top_features, shap_df
