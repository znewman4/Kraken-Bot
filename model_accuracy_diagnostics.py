#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from src.backtesting.runner import run_backtest

def main():
    parser = argparse.ArgumentParser(
        description='Run prediction vs. realized-return diagnostics'
    )
    parser.add_argument(
        '-c', '--config',
        default='config.yml',
        help='Path to backtest config file'
    )
    args = parser.parse_args()

    # Run backtest to get metrics
    metrics, stats, cerebro = run_backtest(args.config)

    # Prepare DataFrame
    df = metrics.copy()
    df['real_ret'] = df['close'].pct_change().shift(-1)
    df = df.dropna(subset=['exp_return', 'real_ret'])

    # Correlation
    corr = df['exp_return'].corr(df['real_ret'])
    # Directional accuracy
    hits = np.sign(df['exp_return']) == np.sign(df['real_ret'])
    hit_rate = hits.mean()

    # Print results
    print(f"Prediction vs. next-bar return corr: {corr:.3f}")
    print(f"Directional accuracy: {hit_rate:.1%} ({hits.sum()}/{len(hits)})")

if __name__ == '__main__':
    main()
