import pandas as pd
import matplotlib.pyplot as plt

# Load saved log
df = pd.read_csv("trade_log.csv")
print(f"Loaded {len(df)} trades")

# Basic summary
print(df.describe())

# PnL distribution
plt.hist(df['net_pnl'], bins=50)
plt.title("Net PnL Distribution")
plt.xlabel("Net PnL")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Scatter: Predicted vs Actual
plt.scatter(df['predicted_exp_r'], df['net_pnl'], alpha=0.5)
plt.title("Predicted Return vs Net PnL")
plt.xlabel("Predicted exp_r")
plt.ylabel("Net PnL")
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.grid(True)
plt.show()

# Edge vs PnL
plt.scatter(df['edge'], df['net_pnl'], alpha=0.5)
plt.title("Edge vs Net PnL")
plt.xlabel("Edge (exp_r / vol)")
plt.ylabel("Net PnL")
plt.grid(True)
plt.show()

# Trade size vs PnL
plt.scatter(df['position_size'], df['net_pnl'], alpha=0.5)
plt.title("Position Size vs Net PnL")
plt.xlabel("Position Size")
plt.ylabel("Net PnL")
plt.grid(True)
plt.show()

# Hold time vs PnL
plt.scatter(df['hold_time_mins'], df['net_pnl'], alpha=0.5)
plt.title("Hold Time vs Net PnL")
plt.xlabel("Hold time (mins)")
plt.ylabel("Net PnL")
plt.grid(True)
plt.show()

# Bucket predictions into deciles
df['bucket'] = pd.qcut(df['predicted_exp_r'], 10, labels=False)

# Mean actual PnL per bucket
bucket_stats = df.groupby('bucket')['net_pnl'].mean()

bucket_stats.plot(kind='bar')
plt.title("Calibration: Avg Net PnL by Predicted Return Decile")
plt.xlabel("Predicted exp_r Bucket (low → high)")
plt.ylabel("Average Net PnL")
plt.grid(True)
plt.show()

