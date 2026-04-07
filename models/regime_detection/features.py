from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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
    "return_1",
    "volatility_24",
    "volatility_72",
    "range_ratio",
    "body_ratio",
    "volume_change",
]


def _load_ticks_for_feature_build(pair_path: Path, latest_only: bool) -> dict[int, pd.DataFrame]:
    if not latest_only:
        yearly = load_tick_data(str(pair_path))
        return preprocess_tick_data(yearly)

    parquet_files = sorted(pair_path.glob("*_combined.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in pair directory: {pair_path}")

    latest_file = max(parquet_files, key=lambda p: int(p.stem.split("_")[0]))
    latest_year = int(latest_file.stem.split("_")[0])
    latest_df = pd.read_parquet(latest_file)

    return preprocess_tick_data({latest_year: latest_df})


def combine_yearly_ticks(data_ticks: dict[int, pd.DataFrame]) -> pd.DataFrame:
    if not data_ticks:
        raise ValueError("No yearly tick data available. Check pair path and parquet files.")

    combined = pd.concat(data_ticks.values(), ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


def add_regime_features(ohlcv_df: pd.DataFrame, return_lag: int = 1) -> pd.DataFrame:
    if return_lag < 1:
        raise ValueError("return_lag must be >= 1")

    df = ohlcv_df.copy()
    df = df.dropna(subset=["open_price", "high_price", "low_price", "close_price"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["return_1"] = np.log(df["close_price"] / df["close_price"].shift(return_lag))
    df["volatility_24"] = df["return_1"].rolling(24).std()
    df["volatility_72"] = df["return_1"].rolling(72).std()

    close_safe = df["close_price"].replace(0, np.nan)
    open_safe = df["open_price"].replace(0, np.nan)

    df["range_ratio"] = (df["high_price"] - df["low_price"]) / close_safe
    df["body_ratio"] = (df["close_price"] - df["open_price"]) / open_safe
    df["volume_change"] = np.log((df["volume"] + 1) / (df["volume"].shift(1) + 1))

    clean = df.dropna(subset=BASE_OHLCV_COLUMNS + FEATURE_COLUMNS).reset_index(drop=True)
    return clean


def build_feature_table(
    data_root: str | Path,
    pair: str = "xauusd",
    timeframe: str = "1h",
    return_lag: int = 1,
    max_bars: int | None = None,
) -> pd.DataFrame:
    pair_path = Path(data_root) / pair
    preprocessed = _load_ticks_for_feature_build(pair_path, latest_only=max_bars is not None)
    ticks = combine_yearly_ticks(preprocessed)

    ohlcv = calculate_ohlcv(ticks, freq=timeframe)
    features = add_regime_features(ohlcv, return_lag=return_lag)

    if max_bars is not None:
        features = features.tail(max_bars).reset_index(drop=True)

    if features.empty:
        raise ValueError("Feature table is empty after preprocessing.")

    return features
