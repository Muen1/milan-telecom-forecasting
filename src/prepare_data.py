import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_series(square_id, target_df, test_start='2013-12-16', test_end='2013-12-22 23:50:00'):
    """Extract, align to 10-min frequency, and scale series for one square."""
    df = target_df[target_df['square_id'] == square_id].copy()
    # Remove duplicate datetime for the same square (keep first occurrence)
    df = df.drop_duplicates(subset='datetime', keep='first')
    df = df.sort_values('datetime')
    # Set index and resample to 10-min frequency
    df = df.set_index('datetime').asfreq('10min')
    df['traffic'] = df['traffic'].interpolate(method='linear').ffill().bfill()
    # Split into train (before test start) and test (forecast week)
    train = df.loc[:test_start].iloc[:-1]   # all up to but not including first test point
    test = df.loc[test_start:test_end]
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"Insufficient data for square {square_id}: train={len(train)}, test={len(test)}")
    scaler = MinMaxScaler(feature_range=(0,1))
    train_scaled = scaler.fit_transform(train[['traffic']]).flatten()
    test_scaled = scaler.transform(test[['traffic']]).flatten()
    return train_scaled, test_scaled, scaler, train.index, test.index
