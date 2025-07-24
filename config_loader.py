import os, yaml, json


from dotenv import load_dotenv
load_dotenv()


# 1. sensible defaults
DEFAULTS = {
    'data': {'raw_data_path': 'data/raw/', 'cleaned_data_path': 'data/cleaned/', 'feature_data_path': 'data/features/'},
    'backtest': {'commission': 0.0025, 'cash': 10000},
    # add more defaults here as you see fit...
}

def merge_dicts(d, o):
    res = d.copy()
    for k, v in o.items():
        if k in res and isinstance(res[k], dict) and isinstance(v, dict):
            res[k] = merge_dicts(res[k], v)
        else:
            res[k] = v
    return res

def _validate(cfg):
    required = [('exchange.api_key', "You must set exchange.api_key"),
                ('exchange.api_secret',  "You must set exchange.secret"),
                ('model.type',       "You must set model.type")]
    for path, msg in required:
        parts = path.split('.')
        d = cfg
        for p in parts:
            if p not in d or d[p] is None:
                raise KeyError(f"Config error: {msg} (missing '{path}')")
            d = d[p]

def load_config(path='config.yml', overrides=None):
    # A. read YAML or JSON
    with open(path) as f:
        cfg = json.load(f) if path.endswith('.json') else yaml.safe_load(f)
    # B. merge defaults
    cfg = merge_dicts(DEFAULTS, cfg)
    # C. resolve env vars
    def resolve(v):
        if isinstance(v, str) and v.startswith('${') and v.endswith('}'):
            return os.getenv(v[2:-1])
        return v
    def walk(d):
        for k, v in d.items():
            if isinstance(v, dict): walk(v)
            else:                  d[k] = resolve(v)
    walk(cfg)
    # D. apply CLI overrides like ["backtest.start_date=2022-01-01"]
    if overrides:
        for ov in overrides:
            key, val = ov.split('=', 1)
            parts, sub = key.split('.'), cfg
            for p in parts[:-1]:
                sub = sub.setdefault(p, {})
            try:    sub[parts[-1]] = json.loads(val)
            except: sub[parts[-1]] = val
    # E. verify required fields
    _validate(cfg)
    return cfg
