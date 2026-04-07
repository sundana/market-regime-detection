from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


DEFAULT_COLORS = {
    "Bullish": "rgba(30, 136, 229, 0.25)",
    "Bearish": "rgba(229, 57, 53, 0.25)",
    "Sideways": "rgba(158, 158, 158, 0.20)",
    "Volatile": "rgba(251, 140, 0, 0.22)",
    "Unknown": "rgba(120, 120, 120, 0.18)",
}


def _collect_segments(df: pd.DataFrame, label_col: str) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    segments: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    if df.empty:
        return segments

    start_idx = 0
    labels = df[label_col].tolist()

    for i in range(1, len(df)):
        if labels[i] != labels[i - 1]:
            segments.append((df.iloc[start_idx]["timestamp"], df.iloc[i - 1]["timestamp"], labels[i - 1]))
            start_idx = i

    segments.append((df.iloc[start_idx]["timestamp"], df.iloc[-1]["timestamp"], labels[-1]))
    return segments


def plot_candlestick_with_regimes(
    df: pd.DataFrame,
    label_col: str,
    output_path: str | Path,
    title: str,
    color_map: dict[str, str] | None = None,
) -> Path:
    if df.empty:
        raise ValueError("Cannot build candlestick plot from empty dataframe.")

    palette = color_map or DEFAULT_COLORS
    clean = df.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"])

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

    for x0, x1, regime in _collect_segments(clean, label_col):
        fill_color = palette.get(regime, palette["Unknown"])
        fig.add_vrect(x0=x0, x1=x1, fillcolor=fill_color, opacity=1.0, line_width=0, layer="below")

    unique_regimes = [r for r in clean[label_col].dropna().unique().tolist()]
    for regime in unique_regimes:
        fig.add_trace(
            go.Scatter(
                x=[clean["timestamp"].iloc[0]],
                y=[clean["high_price"].max()],
                mode="markers",
                marker={"size": 9, "color": palette.get(regime, palette["Unknown"])},
                name=f"Regime: {regime}",
                visible="legendonly",
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title,
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
