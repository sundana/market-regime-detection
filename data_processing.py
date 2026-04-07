import numpy as np
import pandas as pd


def calculate_ohlcv(df, freq='1h'):
    df_copy = df.copy()
    df_copy['mid_price'] = (df_copy['bid'] + df_copy['ask']) / 2
    df_copy.set_index('timestamp', inplace=True)

    ohlcv = df_copy['mid_price'].resample(freq).ohlc().rename(columns={
        'open': 'open_price',
        'high': 'high_price',
        'low': 'low_price',
        'close': 'close_price'
    })

    volume = df_copy.groupby(pd.Grouper(freq=freq)).size()
    ohlcv['volume'] = volume

    ohlcv.reset_index(inplace=True)

    return ohlcv



def calculate_log_return(df):
    df_copy = df.copy()
    df_copy['log_return'] = np.log(df_copy['close_price'] / df_copy['close_price'].shift(2))
    df_copy.dropna(inplace=True)
    return df_copy