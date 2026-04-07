from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .detectors import GMMDetector, HMMDetector, KMeansDetector
from .evaluation import add_composite_score, evaluate_model, split_train_test
from .features import FEATURE_COLUMNS, build_feature_table
from .labeling import apply_regime_labels, infer_regime_mapping, summarize_states
from .visualization import plot_candlestick_with_regimes


def _build_detectors(n_states: int, seed: int) -> dict[str, object]:
    return {
        "hmm": HMMDetector(n_states=n_states, random_state=seed),
        "gmm": GMMDetector(n_states=n_states, random_state=seed),
        "kmeans": KMeansDetector(n_states=n_states, random_state=seed),
    }


def _run_single_model(
    model_name: str,
    detector: object,
    features_df: pd.DataFrame,
    train_idx: int,
    run_dir: Path,
) -> dict[str, float]:
    state_col = f"{model_name}_state"
    label_col = f"{model_name}_regime"

    x_train = features_df.iloc[:train_idx][FEATURE_COLUMNS].to_numpy()
    x_all = features_df[FEATURE_COLUMNS].to_numpy()

    detector.fit(x_train)
    all_states = detector.predict(x_all)

    predicted_df = features_df.copy()
    predicted_df[state_col] = all_states

    train_pred = predicted_df.iloc[:train_idx].copy()
    test_pred = predicted_df.iloc[train_idx:].copy()

    summary = summarize_states(train_pred, state_col=state_col)
    mapping = infer_regime_mapping(summary, state_col=state_col)

    predicted_df = apply_regime_labels(predicted_df, state_col=state_col, mapping=mapping, label_col=label_col)
    test_labeled = predicted_df.iloc[train_idx:].copy()

    model_dir = run_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    predicted_df.to_csv(model_dir / f"{model_name}_labels.csv", index=False)

    summary["regime_label"] = summary[state_col].map(mapping).fillna("Unknown")
    summary.to_csv(model_dir / f"{model_name}_state_summary.csv", index=False)

    metrics = evaluate_model(test_labeled, state_col=state_col, label_col=label_col)

    chart_path = plot_candlestick_with_regimes(
        test_labeled,
        label_col=label_col,
        output_path=model_dir / f"{model_name}_candlestick_regime.html",
        title=f"Regime Labeling on Candlestick - {model_name.upper()}",
    )
    metrics["chart_path"] = str(chart_path)

    return metrics


def run_experiment(
    data_root: str | Path,
    pair: str = "xauusd",
    timeframe: str = "1h",
    output_root: str | Path = "results",
    n_states: int = 3,
    train_ratio: float = 0.7,
    max_bars: int | None = None,
    seed: int = 42,
) -> Path:
    features_df = build_feature_table(
        data_root=data_root,
        pair=pair,
        timeframe=timeframe,
        return_lag=1,
        max_bars=max_bars,
    )

    _, _, train_idx = split_train_test(features_df, train_ratio=train_ratio)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / "regime_detection" / f"{pair}_{timeframe}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    features_df.to_csv(run_dir / "feature_table.csv", index=False)

    all_metrics: dict[str, dict[str, float]] = {}
    detectors = _build_detectors(n_states=n_states, seed=seed)

    for name, detector in detectors.items():
        all_metrics[name] = _run_single_model(
            model_name=name,
            detector=detector,
            features_df=features_df,
            train_idx=train_idx,
            run_dir=run_dir,
        )

    leaderboard = pd.DataFrame.from_dict(all_metrics, orient="index").reset_index().rename(columns={"index": "model"})
    leaderboard = add_composite_score(leaderboard)
    leaderboard.to_csv(run_dir / "leaderboard.csv", index=False)

    summary_rows = [
        {
            "pair": pair,
            "timeframe": timeframe,
            "n_states": n_states,
            "train_ratio": train_ratio,
            "total_samples": len(features_df),
            "train_samples": train_idx,
            "test_samples": len(features_df) - train_idx,
            "run_dir": str(run_dir),
        }
    ]
    pd.DataFrame(summary_rows).to_csv(run_dir / "run_summary.csv", index=False)

    return run_dir
