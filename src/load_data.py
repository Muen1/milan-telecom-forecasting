import pandas as pd
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import time

DATA_RAW = "data/raw/"
DATA_PROCESSED = "data/processed/"
TARGET_IDS = [4159, 4556]          # initial two squares

def read_and_aggregate_daily_file(filepath):
    # Read columns: square_id (0), timestamp (1), traffic (3)
    df = pd.read_csv(filepath, sep='\s+', header=None, usecols=[0,1,3],
                     names=['square_id', 'timestamp', 'traffic'],
                     dtype={'square_id': 'int32', 'timestamp': 'int64', 'traffic': 'float32'})
    # Convert milliseconds to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    # Aggregate: sum traffic for each square and datetime
    df_agg = df.groupby(['square_id', 'datetime'], as_index=False)['traffic'].sum()
    return df_agg[['square_id', 'datetime', 'traffic']]

def process_all_files():
    files = sorted(glob(os.path.join(DATA_RAW, "*.txt")))
    print(f"Found {len(files)} daily files.")

    total_traffic = np.zeros(10000, dtype=np.float64)
    target_dfs = []

    start_time = time.time()

    for fp in tqdm(files, desc="Processing daily files"):
        df_agg = read_and_aggregate_daily_file(fp)

        # Update total traffic per square (sum of aggregated traffic in this file)
        daily_sum = df_agg.groupby('square_id')['traffic'].sum()
        for sid, val in daily_sum.items():
            total_traffic[sid-1] += val

        # Keep rows for the initial target squares
        mask = df_agg['square_id'].isin(TARGET_IDS)
        if mask.any():
            target_dfs.append(df_agg[mask].copy())

    # Find square with highest total traffic
    highest_id = np.argmax(total_traffic) + 1
    print(f"Square with highest total traffic: {highest_id} (total = {total_traffic[highest_id-1]:.2f})")

    # If highest_id not already in TARGET_IDS, read its aggregated time series separately
    if highest_id not in TARGET_IDS:
        print(f"Reading files for highest traffic square {highest_id}...")
        highest_dfs = []
        for fp in tqdm(files, desc="Reading highest square"):
            df_agg = read_and_aggregate_daily_file(fp)
            sub = df_agg[df_agg['square_id'] == highest_id]
            if not sub.empty:
                highest_dfs.append(sub)
        if highest_dfs:
            target_dfs.append(pd.concat(highest_dfs, ignore_index=True))
            TARGET_IDS.append(highest_id)

    # Combine all target data
    full_target = pd.concat(target_dfs, ignore_index=True)
    full_target = full_target.sort_values(['square_id', 'datetime'])

    # Save to processed folder
    full_target.to_csv(DATA_PROCESSED + "target_series.csv", index=False)
    np.save(DATA_PROCESSED + "total_traffic.npy", total_traffic)

    end_time = time.time()
    print(f"Processing time: {(end_time - start_time)/60:.2f} minutes")
    return highest_id, full_target, total_traffic

if __name__ == "__main__":
    highest_id, df_target, total_traffic = process_all_files()
    print("Done. Aggregated data saved to data/processed/")
