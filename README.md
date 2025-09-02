# Project Title & One-Liner

Quant research framework for regime-aware ML strategies on high-frequency crypto data with execution realism and walk-forward evaluation.

# Why this repo

This project demonstrates my ability to combine machine learning with practical risk awareness in a research setting. It reflects an iterative process: moving from naïve classifiers to calibrated regression models, and discovering first-hand how regime shifts impact hyperparameters and model accuracy. The framework is deliberately modest — execution realism and walk-forward retraining are implemented to highlight process over results — and it remains massively improvable. Its purpose is to show that I can reason about model design, risk controls, and research trade-offs in the way a quant team would expect.

# Project structure

Repo layout:

* `config.yml` — global configuration.
* `main.py` — end-to-end training pipeline: feature engineering, model fitting, calibration.
* `run_kraken_strategy.py` — entry point for live-model backtests.
* `src/` — core modules:

  * `backtesting/` — strategies, runners, feeds, analysis.
  * `pipelines/rolling_walkforward.py` — walk-forward retraining and OOS evaluation.
  * `modeling.py`, `training.py`, `tuning.py`, `calibration.py` — model preparation, tuning, and calibration.
  * `data_cleaning.py`, `technical_engineering.py`, `bulk_download.py`, `data_loading.py` — data ingestion and feature engineering.
* `data/` — raw and processed OHLCV, feature sets, and precomputed returns.
* `logs/diagnostics/` — plots and diagnostic outputs.
* `results/` — aggregate performance exports.
* `models/` — trained model artifacts.
* `tests/`, `requirements.txt`, `trade_log.csv`.

# Data layout

Data is organized under `data/`:

* Raw data: `raw/btc_ohlcv_5min_raw.csv`.
* Engineered data: `processed/btc_ohlcv_5min_engineered.csv`.
* Precomputed features and predictions: `features_with_exp_returns.parquet` (and CSV fallback).

Canonical columns include OHLCV, `vwap`, `count`, technical indicators (`ema_10`, `sma_10`, `rsi_14`, Bollinger bands, MACD family, volatility metrics), return lags, and the target `exp_return` (bps). Optional fields include z-scores such as `z_edge`. Index is datetime at 5-minute frequency in UTC.

# Pipelines overview

Two main pipelines:

* Model training pipeline (`main.py` + supporting modules).
* Backtesting pipeline (`src/backtesting/…` + `pipelines/rolling_walkforward.py`).

# Model training pipeline (main.py area)

Inputs

* Engineered feature set from `data/processed/btc_ohlcv_5min_engineered.csv` with train/test splits defined in `config.yml`.

Steps

* Feature augmentation using `technical_engineering`.
* Preparation of features/targets per horizon with `modeling.prepare_features_and_target`.
* Hyperparameter tuning with `tuning.tune_model` using time-series CV.
* Model training via `training.run_training_pipeline`, optionally applying SHAP-based feature pruning.
* Calibration through `calibration.fit_calibrators_for_config` to correct probability estimates.

Outputs

* Horizon-specific model artifacts (`models/*.json`).
* Feature column lists for reproducibility.
* Calibrated predictions exported via `predict_to_csv.py` and converted to Parquet for backtesting.

Notes

* Seeds and thread limits enforced for deterministic reproducibility.

# Backtesting pipeline (backtesting/)

Both strategies implement the same gating and sizing logic. One version consumes precomputed Parquet outputs for high-speed sweeps, the other integrates live model inference for potential deployment. This separation allows fast hyperparameter search and realistic live-mode simulation.

* Precomputed backtest: `runners/runner_precomputed.py` loads Parquet predictions into `strat_precomputed.KrakenStrategy`.
* Live-model backtest: `runners/runner.py` and `runners/runner_slice.py` call `strategy.KrakenStrategy` with on-the-fly XGBoost predictions.
* Rolling walk-forward: `pipelines/rolling_walkforward.py` retrains per segment and backtests on the following segment, logging IC, hit rate, PnL, and SHAP plots.

# Configuration (config.yml)

The configuration file centralizes all experiment parameters. It defines data sources, training windows, trading logic, and execution costs. Strategies are shaped by editing the YAML—switching data windows, adjusting model weights, or toggling calibration—without changing code.

Key fields include:

* `data.*` for paths and train/test ranges.
* `backtest.*` for cash, commission, slippage, and run controls.
* `trading_logic.*` for model paths, thresholds, quantile windows, volatility windows, ATR-based stop/TP, persistence rules, and sizing constraints.
* `calibration.*` for enabling and configuring calibrators.

# Execution realism & risk gates

The strategy layers multiple gates and execution constraints to enforce realism and robustness:

* **Z-threshold gate**: trades only triggered when weighted horizon predictions yield a z-edge above a configurable threshold.
* **EV gate with costs**: requires expected return (in bps) to exceed `cost_bps` (fee + slippage, \~6–8 bps round trip), ensuring trades are only taken when signal strength justifies transaction costs.
* **Quantile filter**: accepts trades only when expected return lies in the extreme quantiles of recent history, suppressing noise and weak signals.
* **ATR-based stop-loss and take-profit**: dynamic levels scaled by volatility, providing regime-adaptive exits.
* **Warmup and sigma floor**: block trades until sufficient history is available and per-bar volatility exceeds a floor.
* **OCO placement delay**: stop/limit children are only armed one bar after entry fill, preventing same-bar stop-outs and improving realism.
* **Timed exit and cooldown**: positions auto-exit after a max hold period, with cooldown bars preventing immediate re-entry.
* **Position sizing**: edge- and volatility-normalized, capped at `max_position` with a minimum trade size enforced.
* **Persistence check**: optional vote buffer requires consistent signal direction across bars, filtering out transient flips.

# Validation & analysis (analysis/)

Analysis tools in `backtesting/analysis/` include:

* `model_accuracy_diagnostics.py` for prediction vs. realized alignment and calibration checks.
* `parameter_sweeper.py` implementing Optuna Bayesian hyperparameter search.
* `hp_grid_report.py`, `horizon_weights.py`, `custom_tuning.py` for reporting and horizon weighting utilities.
* `analyze_trade_log.py` for trade-level diagnostics: PnL, hold times, stop/TP outcomes, and tail analysis.

Diagnostics produced in `logs/diagnostics/` include IC and hit-rate curves, Sharpe vs. turnover tables, walk-forward performance charts, and trade distribution plots. These outputs inform both parameter selection and model validity checks.

# Reproducibility & performance

Determinism is enforced with fixed seeds and thread caps (`OMP_NUM_THREADS`, MKL/BLAS envs). Performance is optimized with Parquet caching, vectorized slicing, and controlled logging for quiet runs. Models are retrained each walk-forward segment and overwrite existing artifacts, mimicking production retraining cycles.

# Quickstart (commands)

Environment setup:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train on a segment:

```
python main.py -c config.yml
```

Backtest out-of-sample (live-model):

```
python run_kraken_strategy.py -c config.yml
```

Walk-forward OOS evaluation:

```
python -m src.pipelines.rolling_walkforward -c config.yml --train-weeks 12 --test-weeks 2 --diag-h 10
```

Hyperparameter sweep:

```
python -m src.backtesting.analysis.parameter_sweeper
```

# Results placeholders

Diagnostic results and plots are stored under `logs/diagnostics/` and include IC curves, Sharpe and PnL summaries, parameter study results, and trade-level outcome distributions.

# Limitations

Known limitations include reliance on BTC/USDT 5-minute data only, sensitivity of hyperparameters to regime shifts, and simplified latency and slippage modeling. While data coverage is strong, exchange-level microstructure is approximated rather than fully modeled. The design generalizes to other assets but has not yet been extended beyond BTC.

# Roadmap / next steps

* Regime-aware parameter switching via config (dynamic horizon weights, z-thresholds).
* Automated retraining triggered by SHAP drift or feature distributional changes.
* Richer slippage models (queue position, order book imbalance, latency injection).
* Multi-asset extension (cross-asset signals, correlation risk management).
* Portfolio overlays (Kelly sizing, drawdown caps, volatility targeting).
* Integration with execution simulators using L2 order book data.
* Continuous walk-forward retraining with CI/CD integration.
* Advanced statistical extensions: e.g., shrinkage estimators for horizon weighting to stabilize across regimes.


