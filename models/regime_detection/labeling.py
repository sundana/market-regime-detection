from __future__ import annotations

import pandas as pd


def summarize_states(
    df: pd.DataFrame,
    state_col: str,
    return_col: str = "return_1",
    vol_col: str = "volatility_24",
) -> pd.DataFrame:
    summary = (
        df.groupby(state_col)
        .agg(
            samples=(state_col, "size"),
            mean_return=(return_col, "mean"),
            mean_volatility=(vol_col, "mean"),
        )
        .reset_index()
        .sort_values("mean_return")
        .reset_index(drop=True)
    )
    return summary


def infer_regime_mapping(summary: pd.DataFrame, state_col: str) -> dict[int, str]:
    if summary.empty:
        return {}

    mapping: dict[int, str] = {}
    states_by_return = summary.sort_values("mean_return")
    state_ids = states_by_return[state_col].tolist()

    if len(state_ids) == 1:
        mapping[int(state_ids[0])] = "Sideways"
        return mapping

    bearish_state = int(state_ids[0])
    bullish_state = int(state_ids[-1])

    mapping[bearish_state] = "Bearish"
    mapping[bullish_state] = "Bullish"

    vol_threshold = summary["mean_volatility"].median()
    for state in state_ids[1:-1]:
        row = summary.loc[summary[state_col] == state].iloc[0]
        mapping[int(state)] = "Volatile" if row["mean_volatility"] >= vol_threshold else "Sideways"

    return mapping


def apply_regime_labels(
    df: pd.DataFrame,
    state_col: str,
    mapping: dict[int, str],
    label_col: str = "regime",
) -> pd.DataFrame:
    labeled = df.copy()
    labeled[label_col] = labeled[state_col].map(mapping).fillna("Unknown")
    return labeled
