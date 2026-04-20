from __future__ import annotations

import argparse
import os
from pathlib import Path

# Prevent Windows MKL + KMeans warning by capping OpenMP threads early in process startup.
if os.name == "nt":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("MKL_NUM_THREADS", "1")

from models.regime_detection import run_experiment


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        return None
    return [int(v) for v in values]


def _parse_str_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return values or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run market regime detector benchmark workflow.")
    parser.add_argument("--pair", type=str, default="xauusd", help="Pair folder name under data/.")
    parser.add_argument("--timeframe", type=str, default="1h", help="Pandas resample timeframe (e.g., 1h).")
    parser.add_argument("--data-dir", type=str, default="data", help="Data root containing pair folders.")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for experiment artifacts.")
    parser.add_argument("--states", type=int, default=3, help="Number of hidden states / clusters.")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Comma-separated model list to run. "
            "Allowed: hmm,hmm_gmm,gmm,kmeans. Use 'all' (or omit) to run all."
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio for time-based split.")
    parser.add_argument("--max-bars", type=int, default=None, help="Use only latest N bars for faster experiments.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=120,
        help="Walk-forward context window size for out-of-sample state prediction.",
    )
    parser.add_argument(
        "--rolling-step",
        type=int,
        default=1,
        help="Walk-forward step size. Must be 1 to avoid look-ahead.",
    )
    parser.add_argument(
        "--inference-mode",
        type=str,
        default=None,
        help=(
            "Inference mode to use for state prediction: pointwise or walk_forward. "
            "If omitted, the pipeline uses model-aware defaults."
        ),
    )
    parser.add_argument(
        "--load-models-from",
        type=str,
        default=None,
        help="Path to previous run directory containing pretrained detector files.",
    )
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Do not save detector artifacts after training.",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip HTML chart rendering for faster evaluation/training runs.",
    )
    parser.add_argument(
        "--chart-full-range",
        action="store_true",
        help="Render chart on full train+test range and mark train/test split.",
    )
    parser.add_argument(
        "--chart-max-bars",
        type=int,
        default=3000,
        help=(
            "Maximum number of bars rendered in HTML chart (for speed). "
            "Use 0 to disable trimming."
        ),
    )

    parser.add_argument("--hmm-auto-tune", action="store_true", help="Enable auto-tuning for HMM.")
    parser.add_argument(
        "--hmm-state-grid",
        type=str,
        default=None,
        help="Comma-separated HMM state candidates, e.g. 2,3,4.",
    )
    parser.add_argument(
        "--hmm-covariance-grid",
        type=str,
        default=None,
        help="Comma-separated covariance candidates, e.g. diag,full.",
    )
    parser.add_argument(
        "--hmm-iter-grid",
        type=str,
        default=None,
        help="Comma-separated iteration candidates, e.g. 200,400,600.",
    )
    parser.add_argument(
        "--hmm-seed-grid",
        type=str,
        default=None,
        help="Comma-separated seed candidates, e.g. 42,53,65.",
    )
    parser.add_argument(
        "--hmm-tune-train-ratio",
        type=float,
        default=0.8,
        help="Internal train ratio for HMM tuning split inside training data.",
    )
    parser.add_argument(
        "--hmm-gmm-mixtures",
        type=int,
        default=2,
        help="Number of Gaussian mixtures per hidden state for HMM-GMM.",
    )
    parser.add_argument(
        "--hmm-gmm-iter",
        type=int,
        default=300,
        help="Maximum EM iterations for HMM-GMM.",
    )
    parser.add_argument(
        "--hmm-gmm-covariance",
        type=str,
        default="diag",
        help="Covariance type for HMM-GMM emissions (diag/full/spherical/tied).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_experiment(
        data_root=Path(args.data_dir),
        pair=args.pair,
        timeframe=args.timeframe,
        output_root=Path(args.output_dir),
        n_states=args.states,
        selected_models=_parse_str_list(args.models),
        train_ratio=args.train_ratio,
        max_bars=args.max_bars,
        seed=args.seed,
        hmm_auto_tune=args.hmm_auto_tune,
        hmm_state_grid=_parse_int_list(args.hmm_state_grid),
        hmm_covariance_grid=_parse_str_list(args.hmm_covariance_grid),
        hmm_iter_grid=_parse_int_list(args.hmm_iter_grid),
        hmm_seed_grid=_parse_int_list(args.hmm_seed_grid),
        hmm_tune_train_ratio=args.hmm_tune_train_ratio,
        hmm_gmm_n_mix=args.hmm_gmm_mixtures,
        hmm_gmm_n_iter=args.hmm_gmm_iter,
        hmm_gmm_covariance_type=args.hmm_gmm_covariance.strip().lower(),
        load_models_from=Path(args.load_models_from) if args.load_models_from else None,
        save_trained_models=not args.no_save_models,
        generate_charts=not args.no_charts,
        chart_include_train=args.chart_full_range,
        chart_max_bars=None if args.chart_max_bars <= 0 else args.chart_max_bars,
        test_rolling_window=args.rolling_window,
        test_prediction_step=args.rolling_step,
        inference_mode=args.inference_mode,
    )
    print(f"Experiment completed. Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
