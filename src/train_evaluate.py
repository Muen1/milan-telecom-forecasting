import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prepare_data import prepare_series
from sarima_model import sarima_forecast
from nn_models import create_sequences, build_lstm, build_gru, train_nn, forecast_nn

os.makedirs("reports/figures", exist_ok=True)

def mape(actual, pred):
    actual = np.array(actual)
    pred = np.array(pred)
    mask = actual > 1e-3
    if np.sum(mask) == 0:
        return 100.0
    return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100

target_df = pd.read_csv("data/processed/target_series.csv", parse_dates=['datetime'])
square_ids = target_df['square_id'].unique()
print(f"Evaluating on squares: {square_ids}")

results = {}
seq_length = 72

for sid in square_ids:
    print(f"\n========== Square {sid} ==========")
    train_scaled, test_scaled, scaler, train_idx, test_idx = prepare_series(sid, target_df)
    
    # ARIMA
    start = time.time()
    sarima_pred_scaled, _ = sarima_forecast(train_scaled, test_scaled)
    train_time_sarima = time.time() - start
    sarima_pred = scaler.inverse_transform(sarima_pred_scaled.reshape(-1,1)).flatten()
    test_actual = scaler.inverse_transform(test_scaled.reshape(-1,1)).flatten()
    
    # LSTM
    X, y = create_sequences(train_scaled, seq_length)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    lstm_model = build_lstm(seq_length, units=40, dropout=0.2)
    start = time.time()
    lstm_model, _ = train_nn(lstm_model, X_train, y_train, X_val, y_val, epochs=30, batch_size=64)
    train_time_lstm = time.time() - start
    last_seq = train_scaled[-seq_length:]
    lstm_pred_scaled = forecast_nn(lstm_model, last_seq, len(test_scaled))
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled.reshape(-1,1)).flatten()
    
    # GRU
    gru_model = build_gru(seq_length, units=40, dropout=0.2)
    start = time.time()
    gru_model, _ = train_nn(gru_model, X_train, y_train, X_val, y_val, epochs=30, batch_size=64)
    train_time_gru = time.time() - start
    gru_pred_scaled = forecast_nn(gru_model, last_seq, len(test_scaled))
    gru_pred = scaler.inverse_transform(gru_pred_scaled.reshape(-1,1)).flatten()
    
    metrics = {
        'ARIMA': {'MAE': mean_absolute_error(test_actual, sarima_pred),
                  'MAPE': mape(test_actual, sarima_pred),
                  'RMSE': np.sqrt(mean_squared_error(test_actual, sarima_pred)),
                  'Train_time_s': train_time_sarima},
        'LSTM': {'MAE': mean_absolute_error(test_actual, lstm_pred),
                 'MAPE': mape(test_actual, lstm_pred),
                 'RMSE': np.sqrt(mean_squared_error(test_actual, lstm_pred)),
                 'Train_time_s': train_time_lstm},
        'GRU': {'MAE': mean_absolute_error(test_actual, gru_pred),
                'MAPE': mape(test_actual, gru_pred),
                'RMSE': np.sqrt(mean_squared_error(test_actual, gru_pred)),
                'Train_time_s': train_time_gru}
    }
    results[sid] = metrics
    
    # Plot
    fig, axes = plt.subplots(3,1, figsize=(12,10), sharex=True)
    models_pred = [sarima_pred, lstm_pred, gru_pred]
    titles = ['ARIMA', 'LSTM', 'GRU']
    for ax, pred, title in zip(axes, models_pred, titles):
        ax.plot(test_idx, test_actual, label='Actual', color='black')
        ax.plot(test_idx, pred, label=title, linestyle='--', alpha=0.8)
        ax.set_ylabel('Traffic')
        ax.legend()
        ax.set_title(f"{title} – Square {sid}")
    axes[-1].set_xlabel('Time (Dec 16-22)')
    plt.tight_layout()
    plt.savefig(f"reports/figures/predictions_square_{sid}.png", dpi=150)
    plt.close()
    
    print(f"ARIMA: MAE={metrics['ARIMA']['MAE']:.2f}, MAPE={metrics['ARIMA']['MAPE']:.2f}%, RMSE={metrics['ARIMA']['RMSE']:.2f}, Train_time={metrics['ARIMA']['Train_time_s']:.2f}s")
    print(f"LSTM: MAE={metrics['LSTM']['MAE']:.2f}, MAPE={metrics['LSTM']['MAPE']:.2f}%, RMSE={metrics['LSTM']['RMSE']:.2f}, Train_time={metrics['LSTM']['Train_time_s']:.2f}s")
    print(f"GRU: MAE={metrics['GRU']['MAE']:.2f}, MAPE={metrics['GRU']['MAPE']:.2f}%, RMSE={metrics['GRU']['RMSE']:.2f}, Train_time={metrics['GRU']['Train_time_s']:.2f}s")

# Save results
rows = []
for sid, model_dict in results.items():
    for model_name, m in model_dict.items():
        rows.append({'Square': sid, 'Model': model_name, **m})
pd.DataFrame(rows).to_csv("reports/model_performance.csv", index=False)
print("\nPerformance summary saved to reports/model_performance.csv")
