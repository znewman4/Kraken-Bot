# src/backtesting/engine/feature_bridge.py
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Iterable
import pandas as pd
import numpy as np


def _normalize_name(name: str) -> str:
    """Lowercase + replace non-word chars (incl '.') with '_' + guard leading digits."""
    s = re.sub(r'\W+', '_', str(name))
    if re.match(r'^\d', s):
        s = '_' + s
    return s.lower()


class FeatureBridge:
    """
    Bridges (model-trained feature names) <-> (Backtrader feed attributes / DataFrame cols).

    Usage:
        bridge = FeatureBridge(bt_data=self_bt_data,
                               models_by_horizon={1: xgb_model, 12: xgb_model, ...},
                               model_paths={ '1': '.../h1.json', '12': '.../h12.json', ...})
        exp_r = bridge.predict_return(weights=[...])  # combined horizon-weighted prediction
    """

    def __init__(self,
                 bt_data,
                 models_by_horizon: Dict[int, object],
                 model_paths: Dict[str, str],
                 verbose: bool = True) -> None:
        self.bt_data = bt_data
        self.df: Optional[pd.DataFrame] = getattr(bt_data.p, 'dataname', None)
        if not isinstance(self.df, pd.DataFrame):
            self.df = None

        self.strict_df = strict_df


        # store references
        self.models = models_by_horizon
        self.model_paths = model_paths

        # per-horizon metadata
        self.trained_feats: Dict[int, List[str]] = {}
        self.attr_map: Dict[int, Dict[str, str]] = {}  # trained_name -> normalized feed attr

        # build maps and validate availability
        for h, model in self.models.items():
            feats = self._load_feature_list(h, model)
            self.trained_feats[h] = feats
            self.attr_map[h] = {f: _normalize_name(f) for f in feats}

        self._preflight_feature_parity(verbose=verbose)

    def predict_return(self, weights):
        dt = self.bt_data.datetime.datetime(0)
        preds = []
        horizons = sorted(self.models.keys())

        for h in horizons:
            feats_in_order = self.trained_feats[h]

            if self.strict_df:
                # === OLD METHOD: use DF ONLY, EXACT TRAINED NAMES/ORDER ===
                if self.df is None:
                    raise RuntimeError("strict_df=True but no DataFrame found on bt_data.p.dataname")
                try:
                    values = [float(self.df.at[dt, f]) for f in feats_in_order]
                except KeyError as e:
                    missing = [f for f in feats_in_order if f not in self.df.columns][:8]
                    raise KeyError(f"[strict_df] Missing feature(s) at {dt}. Example missing: {missing}") from e
            else:
                # === current bridge path (mixed feed/df) ===
                row = self._make_feature_row(h)
                values = [row[f] for f in feats_in_order]

            X = pd.DataFrame([values], columns=feats_in_order)
            pred = float(self.models[h].predict(X)[0])
            preds.append(pred)
            # (optional) quick print to verify scale
            # print(f"[exp_r][{dt}][h={h}] {pred:.8f}")

        w = list(weights)[:len(preds)]
        out = float(sum(preds) / max(len(preds), 1)) if (not w or sum(w) == 0) \
            else float(sum(p * wi for p, wi in zip(preds, w)) / sum(w))
        # print(f"[exp_r][{dt}][combined] {out:.8f}")
        return out

    # ---------- Internals ----------
    def _load_feature_list(self, h: int, model) -> List[str]:
        """
        Prefer a sidecar '<model>.features.json' if it exists; fall back to booster.feature_names.
        """
        import os
        path = self.model_paths.get(str(h))
        if path:
            sidecar = os.path.splitext(path)[0] + "_features.json"
            if os.path.exists(sidecar):
                try:
                    with open(sidecar, "r") as f:
                        feats = json.load(f)
                        if isinstance(feats, list) and feats:
                            return feats
                except Exception:
                    pass

        # fallback to booster names
        feats = getattr(model, "get_booster", lambda: None)()
        if feats is not None:
            names = feats.feature_names or []
            if names:
                return names

        raise RuntimeError(f"Could not obtain trained feature list for horizon {h}")

    def _preflight_feature_parity(self, verbose: bool = True) -> None:
        """
        Ensure every trained feature can be read from either:
        - a Backtrader feed line (normalized attr), or
        - the underlying pandas DataFrame (exact name or dotted->underscored).
        """
        have_attrs = {a for a in dir(self.bt_data) if not a.startswith('_')}
        have_cols = set(self.df.columns) if self.df is not None else set()

        problems = []
        for h, feats in self.trained_feats.items():
            amap = self.attr_map[h]
            for f in feats:
                attr = amap[f]
                if attr in have_attrs:
                    continue
                if f in have_cols:
                    continue
                if f.replace('.', '_') in have_cols:
                    continue
                problems.append((h, f, attr))

        if problems:
            msg = "Feature mismatch (training vs feed/DF). Examples:\n" + \
                  "\n".join(f"  h={h}: trained='{f}' -> feed_attr='{attr}' not found as line or DF col"
                            for h, f, attr in problems[:10])
            raise RuntimeError(msg)

        if verbose:
            print("Feature parity OK — all trained features are available via feed or DF.")

    def _make_feature_row(self, h: int) -> Dict[str, float]:
        """Build a dict with keys EXACTLY matching the model's trained feature names."""
        row = {}
        amap = self.attr_map[h]
        for trained_name, feed_attr in amap.items():
            row[trained_name] = self._read_value(trained_name, feed_attr)
        return row

    def _read_value(self, trained_name: str, feed_attr: str) -> float:
        """
        Order of attempts:
        1) Backtrader line (normalized attr).
        2) pandas DataFrame at current dt using exact trained name.
        3) pandas DataFrame using dotted->underscored alt name.
        """
        # 1) feed attribute
        line = getattr(self.bt_data, feed_attr, None)
        if line is not None:
            try:
                return float(line[0])
            except Exception:
                pass

        # 2/3) DF fallback
        if self.df is not None:
            current_dt = self.bt_data.datetime.datetime(0)
            if trained_name in self.df.columns:
                return float(self.df.at[current_dt, trained_name])
            alt = trained_name.replace('.', '_')
            if alt in self.df.columns:
                return float(self.df.at[current_dt, alt])

        raise KeyError(f"Missing feature '{trained_name}' (mapped attr='{feed_attr}').")
