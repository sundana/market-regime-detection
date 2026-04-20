from __future__ import annotations

from pathlib import Path
import sys

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import load_tick_data, preprocess_tick_data
from data_processing import calculate_ohlcv


BASE_OHLCV_COLUMNS = [
    "timestamp",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
]


FEATURE_COLUMNS = [
    "return_ema_14",
    "atr_14_normalized",
    "volume_change_smooth",
]


def _load_ticks_for_feature_build(pair_path: Path) -> dict[int, pl.DataFrame]:
    yearly = load_tick_data(str(pair_path))
    return preprocess_tick_data(yearly)


def combine_yearly_ticks(data_ticks: dict[int, pl.DataFrame]) -> pl.DataFrame:
    if not data_ticks:
        raise ValueError("No yearly tick data available. Check pair path and parquet files.")

    yearly_frames = [frame for _, frame in sorted(data_ticks.items())]
    combined = pl.concat(yearly_frames, how="vertical_relaxed")
    combined = combined.with_columns(
        pl.col("timestamp").cast(pl.Datetime, strict=False)
    ).drop_nulls(subset=["timestamp"])
    combined = combined.sort("timestamp")
    return combined


def add_regime_features(ohlcv_df: pl.DataFrame, return_lag: int = 1) -> pl.DataFrame:
    if return_lag < 1:
        raise ValueError("return_lag must be >= 1")

    df = (
        ohlcv_df
        .drop_nulls(subset=["open_price", "high_price", "low_price", "close_price"])
        .with_columns(pl.col("timestamp").cast(pl.Datetime, strict=False))
        .drop_nulls(subset=["timestamp"])
        .sort("timestamp")
    )

    df = df.with_columns(
        (pl.col("close_price") / pl.col("close_price").shift(return_lag)).log().alias("return_1")
    )

    df = df.with_columns(
        pl.col("return_1").ewm_mean(span=14, adjust=False).alias("return_ema_14")
    )

    prev_close = pl.col("close_price").shift(1)
    df = df.with_columns(
        pl.max_horizontal(
            pl.col("high_price") - pl.col("low_price"),
            (pl.col("high_price") - prev_close).abs(),
            (pl.col("low_price") - prev_close).abs(),
        ).alias("true_range")
    )

    df = df.with_columns(
        pl.col("true_range").rolling_mean(window_size=14).alias("atr_14")
    )
    df = df.with_columns(
        pl.col("atr_14").ewm_mean(span=5, adjust=False).alias("atr_14_smooth")
    )

    close_safe = pl.when(pl.col("close_price") == 0).then(None).otherwise(pl.col("close_price"))

    df = df.with_columns(
        (pl.col("atr_14_smooth") / close_safe).alias("atr_14_normalized")
    )

    df = df.with_columns(
        (((pl.col("volume") + 1) / (pl.col("volume").shift(1) + 1)).log()).alias("volume_change")
    )
    df = df.with_columns(
        pl.col("volume_change").ewm_mean(span=5, adjust=False).alias("volume_change_smooth")
    )

    clean = df.drop_nulls(subset=BASE_OHLCV_COLUMNS + FEATURE_COLUMNS + ["return_1"])
    return clean


def build_feature_table(
    data_root: str | Path,
    pair: str = "xauusd",
    timeframe: str = "1h",
    return_lag: int = 1,
    max_bars: int | None = None,
) -> pl.DataFrame:
    pair_path = Path(data_root) / pair
    preprocessed = _load_ticks_for_feature_build(pair_path)
    ticks = combine_yearly_ticks(preprocessed)

    ohlcv = calculate_ohlcv(ticks, freq=timeframe)
    features = add_regime_features(ohlcv, return_lag=return_lag)

    if max_bars is not None:
        features = features.tail(max_bars)

    if features.is_empty():
        raise ValueError("Feature table is empty after preprocessing.")

    return features




if __name__ == "__main__":
    x = build_feature_table('data', max_bars=200)
    print(x)