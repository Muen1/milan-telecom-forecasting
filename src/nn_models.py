import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def build_lstm(seq_length, units=40, dropout=0.2):
    model = Sequential([
        LSTM(units, input_shape=(seq_length, 1), return_sequences=False),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_gru(seq_length, units=40, dropout=0.2):
    model = Sequential([
        GRU(units, input_shape=(seq_length, 1), return_sequences=False),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_nn(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=64):
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=0)
    return model, history

def forecast_nn(model, last_seq, steps):
    preds = []
    curr_seq = last_seq.copy()
    for _ in range(steps):
        next_val = model.predict(curr_seq.reshape(1, len(curr_seq), 1), verbose=0)[0,0]
        preds.append(next_val)
        curr_seq = np.roll(curr_seq, -1)
        curr_seq[-1] = next_val
    return np.array(preds)
