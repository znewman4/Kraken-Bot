# src/papertrading/runner.py
"""
Paper-trading runner for Kraken’s demo environment using Backtrader + CCXTStore.

This module mirrors your backtesting runner but connects to live bars and
routes orders against the Kraken Futures demo API. It reuses your existing
config_loader and KrakenStrategy (from src/backtesting/strategy2.py) so all
trading logic stays in one place.

Adjustments based on your config.yml:
- Reads `cfg['exchange']` for API credentials, symbol, interval, timeouts, and retries.
- Uses 5‑minute bars (`interval_minute: 5`) as compression.
- Defaults to a 2‑hour warm‑up (can be modified).

Questions:
- Any preference for a different warm‑up period? 2 hrs ≈ 24 bars at 5 min.
- Do you want to configure the demo endpoint under `exchange` (e.g., add an `endpoint` field)?
"""
import os
import argparse
import logging
from datetime import datetime, timedelta

# Install dependencies before use:
#   pip install ccxt
#   pip install git+https://github.com/Dave-Vallance/bt-ccxt-store.git
import backtrader as bt
from ccxtbt.ccxtstore import CCXTStore  # installed via bt_ccxt_store package

from config_loader import load_config
from src.backtesting.strategy2 import KrakenStrategy


def parse_args():
    parser = argparse.ArgumentParser(description="Run paper-trading on Kraken demo")
    parser.add_argument('-c', '--config', default='config.yml',
                        help='Path to config file')
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Logging setup
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    logging.info("Starting paper-trading with config: %s", args.config)

    # --- Load config and extract exchange settings
    cfg    = load_config(args.config)
    ex_cfg = cfg['exchange']

    api_key      = ex_cfg['api_key']
    api_secret   = ex_cfg['api_secret']
    symbol       = ex_cfg['symbol']            # e.g. 'XBTUSD'
    interval     = ex_cfg.get('interval_minute', 5)
    timeout_sec  = ex_cfg.get('timeout', 30)
    retries      = ex_cfg.get('retries', 3)
    retry_backoff= ex_cfg.get('retry_backoff', 5)
    endpoint     = ex_cfg.get('endpoint', 'https://demo-futures.kraken.com')

    # --- Cerebro & CCXTStore
    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(True)  # allow fractional sizing

    # CCXT config: Kraken demo endpoint + timeouts
    store_config = {
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'timeout': int(timeout_sec * 1000),   # ccxt expects ms
        'urls': {'api': endpoint},
    }

    store = CCXTStore(
        exchange=ex_cfg['name'],
        currency=symbol[-3:],                  # last 3 chars, e.g. 'USD'
        config=store_config,
        retries=retries,
        retry_backoff=retry_backoff,
        debug=False
    )
    broker = store.getbroker()
    cerebro.setbroker(broker)

    # --- Data feed: warm up with recent bars (default 2h)
    warmup_hours = 2
    start_time   = datetime.utcnow() - timedelta(hours=warmup_hours)
    data = store.getdata(
        dataname=symbol,
        timeframe=bt.TimeFrame.Minutes,
        compression=interval,
        fromdate=start_time,
        ohlcv_limit=int(warmup_hours * 60 / interval)
    )
    cerebro.adddata(data)

    # --- Add your strategy (uses existing KrakenStrategy)
    cerebro.addstrategy(KrakenStrategy, config=cfg)

    # --- Enter live loop
    logging.info("Entering live run loop...")
    cerebro.run()

if __name__ == '__main__':
    main()
