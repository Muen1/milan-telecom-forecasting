import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
import json

os.makedirs("reports/figures", exist_ok=True)

# Load data
target_df = pd.read_csv("data/processed/target_series.csv", parse_dates=['datetime'])
total_traffic = np.load("data/processed/total_traffic.npy")

# 1. PDF
plt.figure(figsize=(10,5))
sns.histplot(total_traffic, kde=True, stat='density', bins=50)
plt.title("Probability Density Function of Total Traffic per Square (2 months)")
plt.xlabel("Total Internet traffic (CDR count)")
plt.ylabel("Density")
plt.savefig("reports/figures/pdf_total_traffic.png", dpi=150)
plt.close()

# 2. Time series first two weeks (all three squares)
square_ids = target_df['square_id'].unique()
start = pd.Timestamp('2013-11-01')
end = pd.Timestamp('2013-11-14 23:50:00')
fig, axes = plt.subplots(len(square_ids), 1, figsize=(12, 10), sharex=True)
if len(square_ids) == 1:
    axes = [axes]
for i, sid in enumerate(square_ids):
    sub = target_df[target_df['square_id'] == sid]
    sub = sub[(sub['datetime'] >= start) & (sub['datetime'] <= end)]
    axes[i].plot(sub['datetime'], sub['traffic'], linewidth=0.8)
    axes[i].set_title(f"Square {sid}")
    axes[i].set_ylabel("Traffic")
axes[-1].set_xlabel("Time (Nov 1-14, 2013)")
plt.tight_layout()
plt.savefig("reports/figures/timeseries_first_two_weeks.png", dpi=150)
plt.close()

# 3. Decomposition (use highest traffic square)
highest_id = np.argmax(total_traffic) + 1
series = target_df[target_df['square_id'] == highest_id]['traffic'].values
if len(series) >= 288:
    decomp = seasonal_decompose(series, model='additive', period=144)
    fig = decomp.plot()
    fig.set_size_inches(12, 8)
    plt.savefig("reports/figures/decomposition.png", dpi=150)
    plt.close()
else:
    print("Skipping decomposition: insufficient data")

# 4. ACF/PACF
if len(series) > 200:
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
    plot_acf(series, lags=200, ax=ax1)
    plot_pacf(series, lags=200, ax=ax2, method='ywm')
    plt.savefig("reports/figures/acf_pacf.png", dpi=150)
    plt.close()
else:
    print("Skipping ACF/PACF")

# 5. Spatial heatmap – computed regular grid from square_id
grid = np.zeros((100,100))
for sid in range(1, 10001):
    row = (sid-1)//100
    col = (sid-1)%100
    grid[row, col] = total_traffic[sid-1]
plt.figure(figsize=(10,8))
sns.heatmap(grid, cmap='Reds', cbar_kws={'label': 'Total traffic'})
plt.title("Spatial distribution of mobile traffic (Milan grid)")
plt.xlabel("X coordinate (0-99)")
plt.ylabel("Y coordinate (0-99)")
plt.savefig("reports/figures/spatial_heatmap.png", dpi=150)
plt.close()

# 6. Anomalies
mean = series.mean()
std = series.std()
anomalies = np.where(series > mean + 3*std)[0]
plt.figure(figsize=(12,5))
plt.plot(series, alpha=0.7)
plt.scatter(anomalies, series[anomalies], color='red', s=30, label='Anomaly')
plt.title(f"Traffic time series with anomalies (3-sigma) – Square {highest_id}")
plt.legend()
plt.savefig("reports/figures/anomalies.png", dpi=150)
plt.close()

print("All Task 2 figures saved to reports/figures/")
