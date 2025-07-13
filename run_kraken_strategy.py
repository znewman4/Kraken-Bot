from src.backtesting.runner import run_backtest

if __name__ == "__main__":
    print("🔁 Running KrakenStrategy backtest only...")
    metrics, _ = run_backtest("config.yml")
    print("✅ Done.")

    if metrics.empty:
        print("⚠️ No metrics returned. Possibly no trades were made.")
    else:
        final_equity = metrics['pnl'].cumsum().iloc[-1] + metrics['close'].iloc[0]
        total_pnl = metrics['pnl'].sum()
        num_trades = (metrics['signal'] != 0).sum()
        sharpe = metrics['pnl'].mean() / (metrics['pnl'].std() + 1e-8) * (252**0.5)

        print("\n📊 KrakenStrategy Summary:")
        print(f"• Final Equity: ${final_equity:.2f}")
        print(f"• Total PnL:    ${total_pnl:.2f}")
        print(f"• # Trades:     {num_trades}")
        print(f"• Sharpe Ratio: {sharpe:.2f}")

    
