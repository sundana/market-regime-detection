from __future__ import annotations

import numpy as np
import pandas as pd


def split_train_test(df: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if not 0.1 <= train_ratio <= 0.95:
        raise ValueError("train_ratio must be between 0.1 and 0.95")

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced an empty partition.")

    return train_df, test_df, split_idx


def compute_stability_metrics(state_series: pd.Series) -> dict[str, float]:
    states = state_series.to_numpy()
    n = len(states)

    if n < 2:
        return {
            "avg_regime_duration": float("nan"),
            "transitions_per_100": float("nan"),
            "persistence_ratio": float("nan"),
        }

    transitions = np.sum(states[1:] != states[:-1])
    persistence_ratio = float(np.mean(states[1:] == states[:-1]))

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
        "avg_regime_duration": float(np.mean(segment_lengths)),
        "transitions_per_100": float((transitions / (n - 1)) * 100),
        "persistence_ratio": persistence_ratio,
    }


def compute_economic_validity(
    df: pd.DataFrame,
    state_col: str,
    return_col: str = "return_1",
    vol_col: str = "volatility_24",
) -> dict[str, float]:
    state_stats = (
        df.groupby(state_col)
        .agg(mean_return=(return_col, "mean"), mean_vol=(vol_col, "mean"))
        .reset_index()
    )

    overall_return_std = float(df[return_col].std())
    overall_vol_std = float(df[vol_col].std())

    return_sep = float(state_stats["mean_return"].std() / overall_return_std) if overall_return_std > 0 else 0.0
    vol_sep = float(state_stats["mean_vol"].std() / overall_vol_std) if overall_vol_std > 0 else 0.0

    high_vol_state = int(state_stats.loc[state_stats["mean_vol"].idxmax(), state_col])
    spike_threshold = float(df[vol_col].quantile(0.8))
    spikes = df[vol_col] >= spike_threshold

    if int(spikes.sum()) == 0:
        spike_alignment = 0.0
    else:
        spike_alignment = float((df.loc[spikes, state_col] == high_vol_state).mean())

    return {
        "return_separation": return_sep,
        "volatility_separation": vol_sep,
        "vol_spike_alignment": spike_alignment,
    }


def evaluate_model(
    df: pd.DataFrame,
    state_col: str,
    label_col: str = "regime",
    return_col: str = "return_1",
    vol_col: str = "volatility_24",
) -> dict[str, float]:
    stability = compute_stability_metrics(df[state_col])
    economic = compute_economic_validity(df, state_col=state_col, return_col=return_col, vol_col=vol_col)

    regime_count = int(df[label_col].nunique())
    state_count = int(df[state_col].nunique())

    metrics: dict[str, float] = {
        **stability,
        **economic,
        "regime_count": float(regime_count),
        "state_count": float(state_count),
        "test_samples": float(len(df)),
    }
    return metrics


def add_composite_score(leaderboard: pd.DataFrame) -> pd.DataFrame:
    score_df = leaderboard.copy()

    cols = ["persistence_ratio", "vol_spike_alignment", "volatility_separation"]
    for col in cols:
        col_min = score_df[col].min()
        col_max = score_df[col].max()
        if col_max > col_min:
            score_df[f"{col}_norm"] = (score_df[col] - col_min) / (col_max - col_min)
        else:
            score_df[f"{col}_norm"] = 1.0

    score_df["transitions_penalty"] = score_df["transitions_per_100"]
    t_min = score_df["transitions_penalty"].min()
    t_max = score_df["transitions_penalty"].max()
    if t_max > t_min:
        score_df["transitions_penalty_norm"] = (score_df["transitions_penalty"] - t_min) / (t_max - t_min)
    else:
        score_df["transitions_penalty_norm"] = 0.0

    score_df["composite_score"] = (
        0.35 * score_df["persistence_ratio_norm"]
        + 0.35 * score_df["vol_spike_alignment_norm"]
        + 0.30 * score_df["volatility_separation_norm"]
        - 0.15 * score_df["transitions_penalty_norm"]
    )

    return score_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
