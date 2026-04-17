from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


DEFAULT_COLORS = {
    "Bullish": "rgba(46, 125, 50, 0.25)",
    "Bearish": "rgba(229, 57, 53, 0.25)",
    "Sideway": "rgba(158, 158, 158, 0.20)",
    "Sideways": "rgba(158, 158, 158, 0.20)",
    "Volatile": "rgba(251, 140, 0, 0.22)",
    "Unknown": "rgba(120, 120, 120, 0.18)",
}


def _infer_bar_width(df: pd.DataFrame) -> pd.Timedelta:
    if len(df) < 2:
        return pd.Timedelta(hours=1)

    diffs = df["timestamp"].diff().dropna()
    positive_diffs = diffs[diffs > pd.Timedelta(0)]
    if positive_diffs.empty:
        return pd.Timedelta(hours=1)
    return positive_diffs.median()


def _collect_segments(df: pd.DataFrame, label_col: str) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    segments: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    if df.empty:
        return segments

    bar_width = _infer_bar_width(df)
    start_idx = 0
    labels = df[label_col].tolist()

    for i in range(1, len(df)):
        if labels[i] != labels[i - 1]:
            # Use an exclusive end bound (next bar timestamp) so single-bar regimes remain visible.
            segments.append((df.iloc[start_idx]["timestamp"], df.iloc[i]["timestamp"], labels[i - 1]))
            start_idx = i

    final_end = pd.Timestamp(df.iloc[-1]["timestamp"]) + bar_width
    segments.append((df.iloc[start_idx]["timestamp"], final_end, labels[-1]))
    return segments


def plot_candlestick_with_regimes(
    df: pd.DataFrame,
    label_col: str,
    output_path: str | Path,
    title: str,
    color_map: dict[str, str] | None = None,
    inference_note: str | None = None,
    show_rolling_points: bool = True,
    train_test_split_ts: pd.Timestamp | str | None = None,
) -> Path:
    if df.empty:
        raise ValueError("Cannot build candlestick plot from empty dataframe.")

    palette = color_map or DEFAULT_COLORS
    unknown_color = palette.get("Unknown", DEFAULT_COLORS["Unknown"])
    clean = df.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"])
    clean = clean.sort_values("timestamp").reset_index(drop=True)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=clean["timestamp"],
                open=clean["open_price"],
                high=clean["high_price"],
                low=clean["low_price"],
                close=clean["close_price"],
                name="Price",
            )
        ]
    )

    if show_rolling_points and "rolling_window_start" in clean.columns:
        rolling_mask = clean["rolling_window_start"].notna()
        if bool(rolling_mask.any()):
            rolling_meta = clean.loc[rolling_mask, "rolling_window_start"].astype("string").fillna("N/A")
            fig.add_trace(
                go.Scatter(
                    x=clean.loc[rolling_mask, "timestamp"],
                    y=clean.loc[rolling_mask, "close_price"],
                    mode="markers",
                    marker={"size": 4, "color": "rgba(33, 33, 33, 0.45)"},
                    customdata=rolling_meta,
                    name="Rolling prediction point",
                    hovertemplate=(
                        "Time=%{x}<br>Close=%{y:.5f}<br>Window start=%{customdata}<extra></extra>"
                    ),
                    showlegend=True,
                )
            )

    for x0, x1, regime in _collect_segments(clean, label_col):
        fill_color = palette.get(regime, unknown_color)
        fig.add_vrect(x0=x0, x1=x1, fillcolor=fill_color, opacity=1.0, line_width=0, layer="below")

    unique_regimes = [r for r in clean[label_col].dropna().unique().tolist()]
    for regime in unique_regimes:
        fig.add_trace(
            go.Scatter(
                x=[clean["timestamp"].iloc[0]],
                y=[clean["high_price"].max()],
                mode="markers",
                marker={"size": 9, "color": palette.get(regime, unknown_color)},
                name=f"Regime: {regime}",
                visible="legendonly",
                showlegend=True,
            )
        )

    if train_test_split_ts is not None:
        split_ts = pd.Timestamp(train_test_split_ts)
        fig.add_vline(
            x=split_ts,
            line_width=1,
            line_dash="dash",
            line_color="rgba(33, 33, 33, 0.85)",
            opacity=0.85,
        )
        fig.add_annotation(
            x=split_ts,
            y=1.0,
            yref="paper",
            text="Train/Test split",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"size": 11, "color": "rgba(33, 33, 33, 0.90)"},
            bgcolor="rgba(255, 255, 255, 0.70)",
        )

    title_text = title if not inference_note else f"{title}<br><sup>{inference_note}</sup>"

    fig.update_layout(
        title=title_text,
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        height=780,
        width=1400,
        legend_title="Legend",
        xaxis_rangeslider_visible=False,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs="cdn")
    return output
