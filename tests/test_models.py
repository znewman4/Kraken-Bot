# tests/test_models.py
import json, pandas as pd, numpy as np, xgboost as xgb

MODEL = "models/xgb_h1.json"
COLS  = "models/xgb_h1.json.cols.json"
DATA  = "data/processed/btc_ohlcv_5min_engineered.csv" 

# --- Load model + schema ---
m = xgb.XGBRegressor()
m.load_model(MODEL)
b = m.get_booster()

with open(COLS) as f:
    cols = json.load(f)

print(f"num_trees={b.trees_to_dataframe().Tree.nunique()}, base_score={b.attr('base_score')}")
print("feature_names == .cols.json ?", b.feature_names == cols)

# --- Load engineered dataset ---
df = pd.read_csv(DATA, parse_dates=["time"]).set_index("time")
df = df.select_dtypes(include=[np.number])  # only numeric

# Normalize columns like in main.py (lowercase + underscores)
df.columns = [c.lower().replace('.', '_') for c in df.columns]

# --- Feature existence check ---
missing = [c for c in cols if c not in df.columns]
extra   = [c for c in df.columns if c not in cols]

print("\n--- Feature schema check ---")
if missing:
    print(f"❌ Missing in dataset: {missing}")
else:
    print("✅ All .cols.json features found in engineered dataset")

if extra:
    print(f"(Note: dataset has extra columns not in .cols.json: {extra[:10]}{'...' if len(extra)>10 else ''})")

# --- Bulk predictions (sanity) ---
X = df[cols].dropna().astype("float32")
pred = m.predict(X)
print(f"\npred_std={np.std(pred):.6f}, min={np.min(pred):.3f}, max={np.max(pred):.3f}, n={len(pred)}")

# --- Strategy-style row diagnostics ---
def make_feature_row_strategy(df_row, feature_names):
    return np.array([df_row[f] for f in feature_names], dtype=np.float32).reshape(1, -1)

print("\n--- Strategy-style row diagnostics ---")
sample_idx = np.random.choice(X.index, size=5, replace=False)
for i in sample_idx:
    row = make_feature_row_strategy(X.loc[i], cols)
    pred_val = float(m.predict(row)[0])
    zero_frac = (row == 0.0).mean()
    print(f"time={i}, pred={pred_val:.3f}, zero_frac={zero_frac:.2%}, row_head={row.flatten()[:5]}")