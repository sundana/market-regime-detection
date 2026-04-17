from __future__ import annotations

import math
from typing import Any

import polars as pl


def _to_polars_df(df: Any) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df

    if hasattr(df, "columns") and hasattr(df, "iloc"):
        return pl.from_pandas(df)

    try:
        return pl.DataFrame(df)
    except Exception as exc:
        raise TypeError("Unsupported dataframe type for evaluation module.") from exc


def _restore_frame_type(df: pl.DataFrame, template: Any) -> Any:
    if isinstance(template, pl.DataFrame):
        return df

    if hasattr(template, "columns") and hasattr(template, "iloc"):
        return df.to_pandas()

    return df


def split_train_test(df: Any, train_ratio: float = 0.7) -> tuple[Any, Any, int]:
    if not 0.1 <= train_ratio <= 0.95:
        raise ValueError("train_ratio must be between 0.1 and 0.95")

    df_pl = _to_polars_df(df)

    split_idx = int(df_pl.height * train_ratio)
    train_df = df_pl.slice(0, split_idx)
    test_df = df_pl.slice(split_idx)

    if train_df.is_empty() or test_df.is_empty():
        raise ValueError("Train/test split produced an empty partition.")

    return _restore_frame_type(train_df, df), _restore_frame_type(test_df, df), split_idx


def compute_stability_metrics(state_series: Any) -> dict[str, float]:
    if isinstance(state_series, pl.Series):
        states = state_series.to_list()
    elif hasattr(state_series, "to_list"):
        states = state_series.to_list()
    elif hasattr(state_series, "tolist"):
        states = state_series.tolist()
    else:
        states = list(state_series)

    n = len(states)

    if n < 2:
        return {
            "avg_regime_duration": float("nan"),
            "transitions_per_100": float("nan"),
            "persistence_ratio": float("nan"),
        }

    transitions = sum(1 for i in range(1, n) if states[i] != states[i - 1])
    persistence_ratio = float(sum(1 for i in range(1, n) if states[i] == states[i - 1]) / (n - 1))

    segment_lengths = []
    current_len = 1
    for i in range(1, n):
        if states[i] == states[i - 1]:
            current_len += 1
        else:
            segment_lengths.append(current_len)
            current_len = 1
    segment_lengths.append(current_len)

    return {
        "avg_regime_duration": float(sum(segment_lengths) / len(segment_lengths)),
        "transitions_per_100": float((transitions / (n - 1)) * 100),
        "persistence_ratio": persistence_ratio,
    }


def compute_economic_validity(
    df: Any,
    state_col: str,
    return_col: str = "return_1",
    vol_col: str = "volatility_24",
) -> dict[str, float]:
    df_pl = _to_polars_df(df)

    state_stats = df_pl.group_by(state_col).agg([
        pl.col(return_col).mean().alias("mean_return"),
        pl.col(vol_col).mean().alias("mean_vol"),
    ])

    if state_stats.is_empty():
        return {
            "return_separation": 0.0,
            "volatility_separation": 0.0,
            "vol_spike_alignment": 0.0,
        }

    overall_return_std = df_pl.select(pl.col(return_col).std()).item()
    overall_vol_std = df_pl.select(pl.col(vol_col).std()).item()
    state_return_std = state_stats.select(pl.col("mean_return").std()).item()
    state_vol_std = state_stats.select(pl.col("mean_vol").std()).item()

    overall_return_std = float(overall_return_std) if overall_return_std is not None else float("nan")
    overall_vol_std = float(overall_vol_std) if overall_vol_std is not None else float("nan")
    state_return_std = float(state_return_std) if state_return_std is not None else float("nan")
    state_vol_std = float(state_vol_std) if state_vol_std is not None else float("nan")

    if overall_return_std > 0 and not math.isnan(state_return_std):
        return_sep = float(state_return_std / overall_return_std)
    else:
        return_sep = 0.0

    if overall_vol_std > 0 and not math.isnan(state_vol_std):
        vol_sep = float(state_vol_std / overall_vol_std)
    else:
        vol_sep = 0.0

    high_vol_state = (
        state_stats
        .sort("mean_vol", descending=True)
        .select(state_col)
        .head(1)
        .item()
    )

    spike_threshold = df_pl.select(pl.col(vol_col).quantile(0.8)).item()
    if spike_threshold is None:
        spike_alignment = 0.0
    else:
        spike_df = df_pl.filter(pl.col(vol_col) >= float(spike_threshold))
        if spike_df.is_empty():
            spike_alignment = 0.0
        else:
            spike_alignment_raw = spike_df.select(
                (pl.col(state_col) == high_vol_state).cast(pl.Float64).mean()
            ).item()
            spike_alignment = float(spike_alignment_raw) if spike_alignment_raw is not None else 0.0

    return {
        "return_separation": return_sep,
        "volatility_separation": vol_sep,
        "vol_spike_alignment": spike_alignment,
    }


def evaluate_model(
    df: Any,
    state_col: str,
    label_col: str = "regime",
    return_col: str = "return_1",
    vol_col: str = "volatility_24",
) -> dict[str, float]:
    df_pl = _to_polars_df(df)

    stability = compute_stability_metrics(df_pl.get_column(state_col))
    economic = compute_economic_validity(df_pl, state_col=state_col, return_col=return_col, vol_col=vol_col)

    regime_count = int(df_pl.select(pl.col(label_col).n_unique()).item())
    state_count = int(df_pl.select(pl.col(state_col).n_unique()).item())

    metrics: dict[str, float] = {
        **stability,
        **economic,
        "regime_count": float(regime_count),
        "state_count": float(state_count),
        "test_samples": float(df_pl.height),
    }
    return metrics


def add_composite_score(leaderboard: Any) -> Any:
    lb = _to_polars_df(leaderboard)

    cols = ["persistence_ratio", "vol_spike_alignment", "volatility_separation"]
    for col in cols:
        col_min = lb.select(pl.col(col).min()).item()
        col_max = lb.select(pl.col(col).max()).item()

        if col_min is not None and col_max is not None and col_max > col_min:
            lb = lb.with_columns(
                ((pl.col(col) - float(col_min)) / (float(col_max) - float(col_min))).alias(f"{col}_norm")
            )
        else:
            lb = lb.with_columns(pl.lit(1.0).alias(f"{col}_norm"))

    lb = lb.with_columns(pl.col("transitions_per_100").alias("transitions_penalty"))
    t_min = lb.select(pl.col("transitions_penalty").min()).item()
    t_max = lb.select(pl.col("transitions_penalty").max()).item()
    if t_min is not None and t_max is not None and t_max > t_min:
        lb = lb.with_columns(
            (
                (pl.col("transitions_penalty") - float(t_min))
                / (float(t_max) - float(t_min))
            ).alias("transitions_penalty_norm")
        )
    else:
        lb = lb.with_columns(pl.lit(0.0).alias("transitions_penalty_norm"))

    lb = lb.with_columns(
        (
            0.35 * pl.col("persistence_ratio_norm")
            + 0.35 * pl.col("vol_spike_alignment_norm")
            + 0.30 * pl.col("volatility_separation_norm")
            - 0.15 * pl.col("transitions_penalty_norm")
        ).alias("composite_score")
    ).sort("composite_score", descending=True)

    return _restore_frame_type(lb, leaderboard)


def run_single_evaluation_smoke_test() -> None:
    """Run a minimal single-model evaluation smoke test."""
    sample = pl.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:00:00",
                "2026-01-01 01:00:00",
                "2026-01-01 02:00:00",
                "2026-01-01 03:00:00",
                "2026-01-01 04:00:00",
                "2026-01-01 05:00:00",
            ],
            "return_1": [0.0010, 0.0014, -0.0008, 0.0021, -0.0012, 0.0007],
            "volatility_24": [0.0020, 0.0025, 0.0030, 0.0033, 0.0022, 0.0028],
            "hmm_state": [0, 0, 1, 1, 2, 2],
            "hmm_regime": ["Bullish", "Bullish", "Neutral", "Neutral", "Bearish", "Bearish"],
        }
    ).with_columns(
        pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=True)
    )

    metrics = evaluate_model(
        sample,
        state_col="hmm_state",
        label_col="hmm_regime",
        return_col="return_1",
        vol_col="volatility_24",
    )

    required_keys = {
        "avg_regime_duration",
        "transitions_per_100",
        "persistence_ratio",
        "return_separation",
        "volatility_separation",
        "vol_spike_alignment",
        "regime_count",
        "state_count",
        "test_samples",
    }
    missing = required_keys.difference(metrics.keys())
    if missing:
        raise AssertionError(f"Smoke test failed: missing metric keys {sorted(missing)}")

    if int(metrics["state_count"]) != 3:
        raise AssertionError(
            f"Smoke test failed: expected state_count=3, got {metrics['state_count']}"
        )

    if int(metrics["regime_count"]) != 3:
        raise AssertionError(
            f"Smoke test failed: expected regime_count=3, got {metrics['regime_count']}"
        )

    if int(metrics["test_samples"]) != sample.height:
        raise AssertionError(
            f"Smoke test failed: expected test_samples={sample.height}, got {metrics['test_samples']}"
        )

    print("[Smoke Test] single evaluation: PASSED")
    print(metrics)


if __name__ == "__main__":
    run_single_evaluation_smoke_test()
