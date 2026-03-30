import numpy as np
import pandas as pd
import ta


def engineer_ohlc_features(df, vol_window=21, sma_window=50, rsi_window=14):
    """
    Takes an OHLC dataframe and generates stationary features for regime detection.
    """
    print(f"Engineering features: Volatility({vol_window}), SMA({sma_window}), RSI({rsi_window})...")
    data = df.copy()
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Volatility'] = data['Log_Return'].rolling(window=vol_window).std() * np.sqrt(252)
    data['SMA'] = data['Close'].rolling(window=sma_window).mean()
    data['SMA_Distance'] = (data['Close'] / data['SMA']) - 1
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=rsi_window).rsi()
    data = data.drop(columns=['SMA'])
    data_clean = data.dropna()
    print(f"Done. Output shape: {data_clean.shape}")
    return data_clean