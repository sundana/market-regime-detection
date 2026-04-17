import polars as pl


def calculate_ohlcv(df: pl.DataFrame, freq: str = '1h') -> pl.DataFrame:
    """Aggregate tick data to OHLCV candles using Polars dynamic grouping."""
    df_copy = (
        df
        .with_columns([
            pl.col('timestamp').cast(pl.Datetime, strict=False),
            ((pl.col('bid') + pl.col('ask')) / 2).alias('mid_price'),
        ])
        .drop_nulls(subset=['timestamp', 'mid_price'])
        .sort('timestamp')
    )

    ohlcv = (
        df_copy
        .group_by_dynamic('timestamp', every=freq, period=freq, closed='left', label='left')
        .agg([
            pl.col('mid_price').first().alias('open_price'),
            pl.col('mid_price').max().alias('high_price'),
            pl.col('mid_price').min().alias('low_price'),
            pl.col('mid_price').last().alias('close_price'),
            pl.len().alias('volume'),
        ])
        .drop_nulls(subset=['open_price', 'high_price', 'low_price', 'close_price'])
        .sort('timestamp')
    )

    print(f"Total candles: {ohlcv.height}")

    return ohlcv