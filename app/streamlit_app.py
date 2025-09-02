
import os, datetime, math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="LSTM Stock Forecaster", layout="wide")
st.title("📈 LSTM Stock Price Forecaster")

@st.cache_data(show_spinner=False)
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = df[['Close','Volume']].copy()
    df.reset_index(inplace=True)
    df.rename(columns={'Date':'date','Close':'close','Volume':'volume'}, inplace=True)
    return df

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_sequences(features, target, lookback=60):
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        y.append(target[i])
    return np.array(X), np.array(y)

def build_model(input_shape, lr=1e-3):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
    return model

with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Ticker", value="AAPL")
    start  = st.date_input("Start", value=datetime.date(2015,1,1))
    end    = st.date_input("End", value=datetime.date.today())
    lookback = st.slider("Lookback (days)", 30, 200, 60, 5)
    test_ratio = st.slider("Test ratio", 0.05, 0.4, 0.2, 0.05)
    epochs = st.slider("Epochs", 5, 50, 15, 1)
    batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
    rsi_period = st.slider("RSI period", 5, 30, 14, 1)
    sma_short = st.slider("SMA short", 5, 100, 20, 1)
    sma_long  = st.slider("SMA long", 20, 200, 50, 5)
    run = st.button("Train & Predict")

if run:
    df = fetch_data(ticker, start, end)
    if df.empty:
        st.error("No data. Check the inputs.")
        st.stop()

    df['sma_short'] = df['close'].rolling(sma_short).mean()
    df['sma_long']  = df['close'].rolling(sma_long).mean()
    df['rsi']       = compute_rsi(df['close'], rsi_period)
    df = df.dropna().reset_index(drop=True)

    n = len(df)
    test_size = int(n * test_ratio)
    train_df = df.iloc[:-test_size].copy()
    test_df  = df.iloc[-test_size:].copy()

    feature_cols = ['close', 'sma_short', 'sma_long', 'rsi']
    target_col   = 'close'

    X_train_raw = train_df[feature_cols].values
    X_test_raw  = test_df[feature_cols].values

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_x.fit_transform(X_train_raw)
    X_test_scaled  = scaler_x.transform(X_test_raw)

    y_train_scaled = scaler_y.fit_transform(train_df[[target_col]].values)
    y_test_scaled  = scaler_y.transform(test_df[[target_col]].values)

    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, lookback)
    X_test, y_test   = create_sequences(X_test_scaled, y_test_scaled, lookback)

    model = build_model((X_train.shape[1], X_train.shape[2]))
    callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]

    with st.spinner("Training..."):
        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks
        )

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)

    test_index = test_df.iloc[lookback:]['date'].reset_index(drop=True)
    out = pd.DataFrame({'date': test_index, 'actual': y_true.flatten(), 'pred': y_pred.flatten()})

    rmse = math.sqrt(mean_squared_error(out['actual'], out['pred']))
    mae  = mean_absolute_error(out['actual'], out['pred'])
    mape = np.mean(np.abs((out['actual'] - out['pred']) / out['actual'])) * 100

    st.subheader(f"Metrics — RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}%")

    # Plot predictions vs actual
    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(out['date'], out['actual'], label='Actual')
    ax1.plot(out['date'], out['pred'], label='Predicted')
    ax1.set_title(f"{ticker}: Actual vs Predicted Close")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Price"); ax1.legend()
    st.pyplot(fig1)

    # Plot close + MAs
    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(df['date'], df['close'], label='Close')
    ax2.plot(df['date'], df['sma_short'], label=f"SMA{sma_short}")
    ax2.plot(df['date'], df['sma_long'], label=f"SMA{sma_long}")
    ax2.set_title(f"{ticker}: Close & Moving Averages")
    ax2.set_xlabel("Date"); ax2.set_ylabel("Price"); ax2.legend()
    st.pyplot(fig2)

    # Plot RSI
    fig3, ax3 = plt.subplots(figsize=(10,2.8))
    ax3.plot(df['date'], df['rsi'], label='RSI')
    ax3.axhline(70, linestyle='--')
    ax3.axhline(30, linestyle='--')
    ax3.set_title(f"{ticker}: RSI({rsi_period})")
    ax3.set_xlabel("Date"); ax3.set_ylabel("RSI")
    st.pyplot(fig3)

    # Downloads
    st.download_button("Download predictions CSV", out.to_csv(index=False).encode('utf-8'),
                       file_name=f"predictions_{ticker}.csv", mime="text/csv")

    # Save model weights in memory and offer download
    model_path = f"lstm_model_{ticker}.keras"
    model.save(model_path)
    with open(model_path, "rb") as f:
        st.download_button("Download model (.keras)", f, file_name=model_path, mime="application/octet-stream")
