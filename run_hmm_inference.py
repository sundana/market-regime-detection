from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import pandas as pd

from models.regime_detection.detectors import HMMDetector
from models.regime_detection.features import FEATURE_COLUMNS, build_feature_table
from models.regime_detection.labeling import apply_regime_labels, infer_regime_mapping, summarize_states
from models.regime_detection.visualization import plot_candlestick_with_regimes


def _print_progress_line(prefix: str, current: int, total: int, started_at: float) -> None:
    if total <= 0:
        return

    progress = min(max(current / total, 0.0), 1.0)
    width = 28
    filled = int(width * progress)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.perf_counter() - started_at

    print(
        f"\r{prefix} [{bar}] {current}/{total} ({progress * 100:5.1f}%) elapsed {elapsed:6.1f}s",
        end="",
        flush=True,
    )
    if current >= total:
        print("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HMM walk-forward inference for a selected month.")
    parser.add_argument("--detector-path", type=str, required=True, help="Path to hmm_detector.pkl to use.")
    parser.add_argument(
        "--states",
        type=int,
        default=None,
        help="Optional expected number of HMM states for validation against selected detector.",
    )
    parser.add_argument("--pair", type=str, default="xauusd", help="Pair folder name under data/.")
    parser.add_argument("--timeframe", type=str, default="1h", help="Pandas resample timeframe.")
    parser.add_argument("--data-dir", type=str, default="data", help="Data root containing pair folders.")
    parser.add_argument("--output-dir", type=str, default="results/hmm_inference", help="Inference output root.")
    parser.add_argument("--year", type=int, default=2026, help="Target inference year (default: 2026).")
    parser.add_argument("--month", type=int, default=1, help="Target inference month 1-12 (default: 1).")
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=120,
        help="Walk-forward context length in bars.",
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=None,
        help="Optional cap on latest feature bars before month filtering.",
    )
    parser.add_argument("--no-chart", action="store_true", help="Skip HTML chart rendering.")
    return parser.parse_args()


def _month_range(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    if month < 1 or month > 12:
        raise ValueError("month must be between 1 and 12")

    start = pd.Timestamp(year=year, month=month, day=1)
    if month == 12:
        end_exclusive = pd.Timestamp(year=year + 1, month=1, day=1)
    else:
        end_exclusive = pd.Timestamp(year=year, month=month + 1, day=1)
    return start, end_exclusive


def main() -> None:
    args = parse_args()
    detector_path = Path(args.detector_path)
    if not detector_path.exists():
        raise FileNotFoundError(f"Detector file not found: {detector_path}")

    if args.rolling_window < 2:
        raise ValueError("rolling_window must be >= 2")

    started = time.perf_counter()
    start_ts, end_exclusive_ts = _month_range(args.year, args.month)

    print("[HMM Inference] Building feature table...")
    features_df = build_feature_table(
        data_root=Path(args.data_dir),
        pair=args.pair,
        timeframe=args.timeframe,
        return_lag=1,
        max_bars=args.max_bars,
    )
    features_df["timestamp"] = pd.to_datetime(features_df["timestamp"])
    features_df = features_df.sort_values("timestamp").reset_index(drop=True)

    mask = (features_df["timestamp"] >= start_ts) & (features_df["timestamp"] < end_exclusive_ts)
    target_indices = np.flatnonzero(mask.to_numpy())
    if len(target_indices) == 0:
        raise ValueError(
            "No candles found for selected month. "
            "Try increasing --max-bars or remove it to use full history."
        )

    print(
        f"[HMM Inference] Target month: {start_ts.strftime('%Y-%m')} "
        f"bars={len(target_indices)}"
    )

    detector = HMMDetector.load(detector_path)
    if args.states is not None and int(detector.n_states) != int(args.states):
        raise ValueError(
            f"Detector n_states mismatch: detector has {detector.n_states}, but --states={args.states}"
        )

    all_states = np.full(len(features_df), np.nan)
    rolling_window_start: list[pd.Timestamp | None] = [None] * len(features_df)

    print("[HMM Inference] Running walk-forward prediction...")
    prediction_started = time.perf_counter()
    for offset, row_idx in enumerate(target_indices, start=1):
        window_end_idx = int(row_idx) + 1
        window_start_idx = max(0, window_end_idx - args.rolling_window)
        x_window = features_df.iloc[window_start_idx:window_end_idx][FEATURE_COLUMNS].to_numpy()

        pred_state = int(detector.predict(x_window)[-1])
        all_states[row_idx] = pred_state
        rolling_window_start[row_idx] = pd.Timestamp(features_df.iloc[window_start_idx]["timestamp"])

        _print_progress_line("[HMM Inference]", offset, len(target_indices), prediction_started)

    inferred = features_df.loc[mask].copy()
    inferred["hmm_state"] = all_states[mask.to_numpy()].astype(int)
    inferred["rolling_window_start"] = [rolling_window_start[i] for i in target_indices]
    inferred["rolling_window_size"] = float(args.rolling_window)
    inferred["inference_method"] = "walk_forward"

    summary = summarize_states(inferred, state_col="hmm_state")
    mapping = infer_regime_mapping(summary, state_col="hmm_state")
    inferred = apply_regime_labels(inferred, state_col="hmm_state", mapping=mapping, label_col="hmm_regime")
    summary["regime_label"] = summary["hmm_state"].map(mapping).fillna("Unknown")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{args.pair}_{args.timeframe}_{args.year}{args.month:02d}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    labels_path = run_dir / "hmm_inference_labels.csv"
    summary_path = run_dir / "hmm_inference_state_summary.csv"
    inferred.to_csv(labels_path, index=False)
    summary.to_csv(summary_path, index=False)

    chart_path = ""
    if not args.no_chart:
        chart_path = str(
            plot_candlestick_with_regimes(
                inferred,
                label_col="hmm_regime",
                output_path=run_dir / "hmm_inference_candlestick_regime.html",
                title=f"HMM Inference {args.pair.upper()} {args.timeframe} - {start_ts.strftime('%Y-%m')}",
                inference_note=(
                    f"Custom detector={detector_path.name}; walk-forward window={args.rolling_window}"
                ),
            )
        )

    summary_row = {
        "pair": args.pair,
        "timeframe": args.timeframe,
        "year": args.year,
        "month": args.month,
        "detector_n_states": int(detector.n_states),
        "requested_states": args.states if args.states is not None else "",
        "rows_predicted": len(inferred),
        "range_start": str(inferred["timestamp"].min()),
        "range_end": str(inferred["timestamp"].max()),
        "rolling_window": args.rolling_window,
        "detector_path": str(detector_path),
        "labels_path": str(labels_path),
        "state_summary_path": str(summary_path),
        "chart_path": chart_path,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    pd.DataFrame([summary_row]).to_csv(run_dir / "inference_summary.csv", index=False)

    print(f"[HMM Inference] Done. Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
