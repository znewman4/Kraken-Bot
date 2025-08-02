import argparse
import logging
import yaml
from datetime import datetime, timedelta

import backtrader as bt
from ccxtbt.ccxtstore import CCXTStore
from src.backtesting.strategies.strategytest import KrakenStrategy  # or strategy2 if you prefer

def parse_args():
    p = argparse.ArgumentParser(description="Run live paper-trading on Kraken Futures demo")
    p.add_argument(
        "-c", "--config", default="config.yml",
        help="Path to your YAML config"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # — Logging setup —
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    log = logging.getLogger("papertrading")
    log.info("Starting paper-trading with config: %s", args.config)

    # — Load config.yml —
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exch        = cfg["exchange"]
    exchange_id = exch["name2"]      # e.g. "krakenfutures"
    symbol      = exch["symbol2"]    # e.g. "BTC/USD:BTC"
    interval    = exch.get("interval_minute", 5)

    # ←—— USE api_key2/secret2 if present, else fall back to api_key/secret
    api_key     = exch.get("api_key2", exch.get("api_key"))
    api_secret  = exch.get("api_secret2", exch.get("api_secret"))

    retries     = exch.get("retries", 3)
    timeout_s   = exch.get("timeout", 30)

    # — Determine the correct Futures-demo base_api —
    raw_ep = exch.get("endpoint")
    if raw_ep:
        base_api = raw_ep.rstrip("/")
    else:
        base_api = "https://demo-futures.kraken.com/derivatives/api/v3"

    log.info(">>> base_api is set to: %s", base_api)

    # — Build CCXTStore config (strip creds for init) —
    ccxt_cfg = {
        "apiKey":          api_key,
        "secret":          api_secret,
        "enableRateLimit": True,
        "timeout":         int(timeout_s * 1000),
        "urls": {
            "api": {
                "public":  base_api,
                "private": base_api,
            }
        }
    }
    store_cfg = ccxt_cfg.copy()
    store_cfg.pop("apiKey", None)
    store_cfg.pop("secret", None)

    # — Derive quote currency (e.g. "USD") —
    currency = symbol.split("/")[-1] if "/" in symbol else symbol[-3:]

    # — Instantiate store & reattach credentials —
    store = CCXTStore(
        exchange=exchange_id,
        currency=currency,
        config=store_cfg,
        retries=retries,
        debug=False
    )
    store.exchange.apiKey = api_key
    store.exchange.secret = api_secret

    # — Disable CCXT’s automatic “/v3” prefix (we bake it into base_api) —
    store.exchange.version = ""
    store.exchange.urls['api']['public']  = base_api
    store.exchange.urls['api']['private'] = base_api

    # — Load markets and verify our symbol (no dump) —
    markets = store.exchange.load_markets()
    if symbol not in markets:
        log.error("Configured symbol %r not in loaded markets!", symbol)
        log.error("Please pick a valid market key from the Kraken Futures demo.")
        return
    log.info("Trading market: %s", symbol)

    # — Cerebro & broker setup —
    cerebro = bt.Cerebro()
    cerebro.setbroker(store.getbroker())

    # — Warm-up to seed ATR, volatility & persistence buffers —
    tl          = cfg["trading_logic"]
    atr_p       = tl.get("atr_period", 14)
    vol_w       = tl.get("vol_window", 20)
    persist     = tl.get("persistence", 3)
    warmup_bars = max(atr_p, vol_w) + persist

    now_utc  = datetime.utcnow()
    lookback = timedelta(minutes=interval * warmup_bars)
    fromdate = now_utc - lookback

    data = store.getdata(
        dataname           = symbol,
        timeframe          = bt.TimeFrame.Minutes,
        compression        = interval,
        fromdate           = fromdate,
        ohlcv_limit        = warmup_bars,
        fetch_ohlcv_params = {"partial": False},
    )
    cerebro.adddata(data)

    # — Add strategy & live analyzers —
    cerebro.addstrategy(KrakenStrategy, config=cfg)

    cerebro.addanalyzer(
        bt.analyzers.TradeAnalyzer, _name="trades")

   

    # — Run until Ctrl+C, then print live stats —
    strat = None
    try:
        results = cerebro.run()
        strat   = results[0]
    except KeyboardInterrupt:
        log.info("Interrupted—gathering live stats…")
    if not strat and "results" in locals() and results:
        strat = results[0]
    if not strat:
        log.error("No strategy instance available. Exiting.")
        return
 

    final_val = cerebro.broker.getvalue()
  

    print("\n=== Live Paper-Trading Results ===")
    print(f"Final Portfolio Value: ${final_val:,.2f}")
    
    print("==================================\n")

if __name__ == "__main__":
    main()
