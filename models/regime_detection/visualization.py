from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

DEFAULT_COLORS = {
    "Bullish": "#2e7d32",
    "Bearish": "#e53935",
    "Sideway": "#9e9e9e",
    "Sideways": "#9e9e9e",
    "Volatile": "#fb8c00",
    "Unknown": "#787878",
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
        segments.append((df.iloc[start_idx]["timestamp"], df.iloc[i]["timestamp"], labels[i - 1]))
        start_idx = i

    final_end = pd.Timestamp(df.iloc[-1]["timestamp"]) + bar_width
    segments.append((df.iloc[start_idx]["timestamp"], final_end, labels[-1]))
    return segments


def _to_unix_seconds(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).timestamp())

def plot_candlestick_with_regimes(
    df: pd.DataFrame,
    label_col: str,
    output_path: str | Path,
    title: str,
    color_map: dict[str, str] | None = None,
    inference_note: str | None = None,
    show_rolling_points: bool = True,
    max_rolling_markers: int = 1500,
    train_test_split_ts: pd.Timestamp | str | None = None,
) -> Path:
    if df.empty:
        raise ValueError("Cannot build candlestick plot from empty dataframe.")

    palette = color_map or DEFAULT_COLORS
    unknown_color = palette.get("Unknown", DEFAULT_COLORS["Unknown"])
    clean = df.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"])
    clean = clean.sort_values("timestamp").reset_index(drop=True)
    clean[label_col] = clean[label_col].fillna("Unknown")

    candle_data: list[dict[str, float | int | str]] = []
    for row in clean.itertuples(index=False):
        regime_label = getattr(row, label_col)
        color = palette.get(str(regime_label), unknown_color)
        candle_data.append(
            {
                "time": _to_unix_seconds(getattr(row, "timestamp")),
                "open": float(getattr(row, "open_price")),
                "high": float(getattr(row, "high_price")),
                "low": float(getattr(row, "low_price")),
                "close": float(getattr(row, "close_price")),
                "color": color,
                "borderColor": color,
                "wickColor": color,
            }
        )

    markers: list[dict[str, str | int]] = []
    if show_rolling_points and "rolling_window_start" in clean.columns:
        rolling_points = clean.loc[clean["rolling_window_start"].notna(), ["timestamp"]].copy()
        if not rolling_points.empty:
            if max_rolling_markers > 0 and len(rolling_points) > max_rolling_markers:
                step = max(1, math.ceil(len(rolling_points) / max_rolling_markers))
                rolling_points = rolling_points.iloc[::step].copy()
            markers = [
                {
                    "time": _to_unix_seconds(ts),
                    "position": "belowBar",
                    "color": "#263238",
                    "shape": "circle",
                    "text": "R",
                }
                for ts in rolling_points["timestamp"].tolist()
            ]

    split_line_data: list[dict[str, float | int]] = []
    if train_test_split_ts is not None:
        split_ts = pd.Timestamp(train_test_split_ts)
        y_min = float(clean["low_price"].min())
        y_max = float(clean["high_price"].max())
        split_time = _to_unix_seconds(split_ts)
        split_line_data = [
            {"time": split_time, "value": y_min},
            {"time": split_time, "value": y_max},
        ]

    legend_items: list[dict[str, str]] = []
    unique_regimes = [str(r) for r in clean[label_col].dropna().unique().tolist()]
    for regime in unique_regimes:
        legend_items.append({"name": regime, "color": palette.get(regime, unknown_color)})

    regime_segments: list[dict[str, str | int]] = []
    for start_ts, end_ts, regime in _collect_segments(clean, label_col):
      regime_segments.append(
        {
          "startTime": _to_unix_seconds(start_ts),
          "endTime": _to_unix_seconds(end_ts),
          "color": palette.get(str(regime), unknown_color),
          "name": str(regime),
        }
      )

    subtitle = inference_note or ""

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <script src=\"https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js\"></script>
  <style>
    body {{
      margin: 0;
      font-family: Segoe UI, Tahoma, sans-serif;
      background: #f7f9fc;
      color: #1f2937;
    }}
    .wrap {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 12px;
    }}
    .title {{
      margin: 0 0 6px 0;
      font-size: 20px;
      font-weight: 700;
      line-height: 1.2;
    }}
    .subtitle {{
      margin: 0 0 10px 0;
      color: #4b5563;
      font-size: 13px;
    }}
    #chart {{
      display: none;
    }}
    #chart-shell {{
      position: relative;
      width: 100%;
      height: 780px;
      border: 1px solid #d6dce5;
      border-radius: 10px;
      background: #ffffff;
      overflow: hidden;
    }}
    #regime-bg-layer {{
      position: absolute;
      inset: 0;
      z-index: 1;
      pointer-events: none;
    }}
    #chart-canvas {{
      position: absolute;
      inset: 0;
      z-index: 2;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 10px;
      font-size: 12px;
    }}
    .item {{
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 4px 8px;
      border-radius: 999px;
      background: #eef2f7;
    }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      border: 1px solid #cfd8e3;
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1 class=\"title\">{title}</h1>
    <p class=\"subtitle\">{subtitle}</p>
    <div id="chart"></div>
    <div id="chart-shell">
      <div id="regime-bg-layer"></div>
      <div id="chart-canvas"></div>
    </div>
    <div class=\"legend\" id=\"legend\"></div>
  </div>

  <script>
    const candleData = {json.dumps(candle_data)};
    const markers = {json.dumps(markers)};
    const splitLineData = {json.dumps(split_line_data)};
    const legendItems = {json.dumps(legend_items)};
    const regimeSegments = {json.dumps(regime_segments)};

    const shell = document.getElementById('chart-shell');
    const bgLayer = document.getElementById('regime-bg-layer');
    const canvasHost = document.getElementById('chart-canvas');

    function addCandlestickSeriesCompat(chart, options) {{
      if (typeof chart.addCandlestickSeries === 'function') {{
        return chart.addCandlestickSeries(options);
      }}
      if (typeof chart.addSeries === 'function' && LightweightCharts.CandlestickSeries) {{
        return chart.addSeries(LightweightCharts.CandlestickSeries, options);
      }}
      throw new Error('Candlestick series API is not available in this Lightweight Charts build.');
    }}

    function addLineSeriesCompat(chart, options) {{
      if (typeof chart.addLineSeries === 'function') {{
        return chart.addLineSeries(options);
      }}
      if (typeof chart.addSeries === 'function' && LightweightCharts.LineSeries) {{
        return chart.addSeries(LightweightCharts.LineSeries, options);
      }}
      return null;
    }}

    function setMarkersCompat(series, points) {{
      if (!points || points.length === 0) return;
      if (typeof series.setMarkers === 'function') {{
        series.setMarkers(points);
        return;
      }}
      if (typeof LightweightCharts.createSeriesMarkers === 'function') {{
        LightweightCharts.createSeriesMarkers(series, points);
      }}
    }}

    function hexToRgba(hex, alpha) {{
      const raw = String(hex || '#787878').replace('#', '');
      const normalized = raw.length === 3 ? raw.split('').map((c) => c + c).join('') : raw;
      const safe = /^[0-9a-fA-F]{6}$/.test(normalized) ? normalized : '787878';
      const r = parseInt(safe.slice(0, 2), 16);
      const g = parseInt(safe.slice(2, 4), 16);
      const b = parseInt(safe.slice(4, 6), 16);
      return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
    }}

    const chart = LightweightCharts.createChart(canvasHost, {{
        width: canvasHost.clientWidth,
        height: canvasHost.clientHeight,
        layout: {{
            background: {{ type: 'solid', color: 'transparent' }},
            textColor: '#1f2937',
        }},
        grid: {{
            vertLines: {{ color: '#eef2f7' }},
            horzLines: {{ color: '#eef2f7' }},
        }},
        rightPriceScale: {{ borderColor: '#d6dce5' }},
        timeScale: {{ borderColor: '#d6dce5', timeVisible: true, secondsVisible: false }},
        crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
    }});

    const candleSeries = addCandlestickSeriesCompat(chart, {{
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: true,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
      priceFormat: {{ type: 'price', precision: 3, minMove: 0.001 }},
    }});
    candleSeries.setData(candleData);

    setMarkersCompat(candleSeries, markers);

    if (splitLineData.length > 0) {{
      const splitSeries = addLineSeriesCompat(chart, {{
        color: '#111827',
        lineWidth: 1,
        lineStyle: 2,
        lastValueVisible: false,
        priceLineVisible: false,
      }});
      if (splitSeries) {{
        splitSeries.setData(splitLineData);
      }}
    }}

    function renderRegimeBands() {{
      bgLayer.innerHTML = '';
      const timeScale = chart.timeScale();

      for (const segment of regimeSegments) {{
        const x0 = timeScale.timeToCoordinate(segment.startTime);
        const x1 = timeScale.timeToCoordinate(segment.endTime);
        if (x0 === null || x1 === null) continue;

        const left = Math.min(x0, x1);
        const width = Math.max(Math.abs(x1 - x0), 1);
        const band = document.createElement('div');
        band.style.position = 'absolute';
        band.style.top = '0';
        band.style.bottom = '0';
        band.style.left = `${{left}}px`;
        band.style.width = `${{width}}px`;
        band.style.background = hexToRgba(segment.color, 0.16);
        bgLayer.appendChild(band);
      }}
    }}

    const legendRoot = document.getElementById('legend');
    for (const item of legendItems) {{
      const node = document.createElement('div');
      node.className = 'item';
      node.innerHTML = `<span class=\"swatch\" style=\"background:${{item.color}}\"></span><span>${{item.name}}</span>`;
      legendRoot.appendChild(node);
    }}

    chart.timeScale().fitContent();
    renderRegimeBands();

    if (typeof chart.timeScale().subscribeVisibleLogicalRangeChange === 'function') {{
      chart.timeScale().subscribeVisibleLogicalRangeChange(renderRegimeBands);
    }}
    if (typeof chart.timeScale().subscribeVisibleTimeRangeChange === 'function') {{
      chart.timeScale().subscribeVisibleTimeRangeChange(renderRegimeBands);
    }}

    const resizeObserver = new ResizeObserver((entries) => {{
      for (const entry of entries) {{
        chart.applyOptions({{ width: entry.contentRect.width, height: entry.contentRect.height }});
        renderRegimeBands();
      }}
    }});
    resizeObserver.observe(shell);
  </script>
</body>
</html>
"""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output



