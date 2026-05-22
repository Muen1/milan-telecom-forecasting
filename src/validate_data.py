import pandas as pd
import numpy as np
from load_data import read_and_aggregate_daily_file

# 1. Inspect one daily file after aggregation
raw_file = "data/raw/sms-call-internet-mi-2013-11-01.txt"
df_agg = read_and_aggregate_daily_file(raw_file)
print("After aggregation (square_id, datetime, traffic):")
print(df_agg.head(10))
print("\nTraffic statistics from one daily file (aggregated):")
print(df_agg['traffic'].describe())

# 2. Load processed target_series.csv
target = pd.read_csv("data/processed/target_series.csv", parse_dates=['datetime'])
print("\n--- Processed target_series.csv ---")
print("Info:")
print(target.info())
print("\nUnique squares:", sorted(target['square_id'].unique()))
print("\nTraffic statistics (all squares):")
print(target['traffic'].describe())

# 3. Check a specific square, e.g., 4159
sample = target[target['square_id'] == 4159].head(10)
print("\nSample rows for square 4159:")
print(sample)

# 4. Date range
print("\nDate range in target:", target['datetime'].min(), "to", target['datetime'].max())

# 5. Total traffic per square
total = np.load("data/processed/total_traffic.npy")
print("\nTotal traffic per square (from total_traffic.npy):")
print("Min:", total.min(), "Max:", total.max(), "Mean:", total.mean())
highest_idx = np.argmax(total)
highest_square = highest_idx + 1
print(f"Square with highest total traffic: {highest_square} (total = {total[highest_idx]:.2f})")

# 6. Ensure highest square appears in target_series
if highest_square in target['square_id'].values:
    print(f"✓ Square {highest_square} is present in target_series.csv")
else:
    print(f"✗ Square {highest_square} is NOT in target_series.csv – this indicates a problem in load_data.py")

print("\nValidation complete.")
