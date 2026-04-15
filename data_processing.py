import numpy as np
import pandas as pd


def calculate_ohlcv(df: pd.DataFrame, freq='1h') -> pd.DataFrame:
    """Resample ticks to OHLCV."""
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
    print(f"Total candles: {len(ohlcv)}")

    return ohlcv