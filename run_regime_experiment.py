from __future__ import annotations

import argparse
from pathlib import Path

from models.regime_detection import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run market regime detector benchmark workflow.")
    parser.add_argument("--pair", type=str, default="xauusd", help="Pair folder name under data/.")
    parser.add_argument("--timeframe", type=str, default="1h", help="Pandas resample timeframe (e.g., 1h).")
    parser.add_argument("--data-dir", type=str, default="data", help="Data root containing pair folders.")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for experiment artifacts.")
    parser.add_argument("--states", type=int, default=3, help="Number of hidden states / clusters.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio for time-based split.")
    parser.add_argument("--max-bars", type=int, default=None, help="Use only latest N bars for faster experiments.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_experiment(
        data_root=Path(args.data_dir),
        pair=args.pair,
        timeframe=args.timeframe,
        output_root=Path(args.output_dir),
        n_states=args.states,
        train_ratio=args.train_ratio,
        max_bars=args.max_bars,
        seed=args.seed,
    )
    print(f"Experiment completed. Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
