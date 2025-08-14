#run_kraken_strategy.py

import argparse, logging
from config_loader import load_config
from src.backtesting.runners.runnertest import run_backtest

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", default="config.yml")
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Running backtest with config %s", args.config)
    stats, cerebro = run_backtest(args.config)

    logging.info("Final equity: $%.2f", cerebro.broker.getvalue())
    logging.info("Total PnL:     $%.2f", stats["real_pnl"])
    raw = stats

    flat = {
    "sharpe":       raw["sharpe"],
    "max_dd":       raw["drawdown"]["max"]["drawdown"],
    "total_trades": raw["trades"]["total"]["total"],
    "real_pnl":     stats["real_pnl"],
    }
    print(flat)

    
if __name__ == "__main__":
    main()
