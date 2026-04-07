from __future__ import annotations

from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from .detectors import BaseDetector, GMMDetector, HMMDetector, KMeansDetector
from .evaluation import add_composite_score, evaluate_model, split_train_test
from .features import FEATURE_COLUMNS, build_feature_table
from .labeling import apply_regime_labels, infer_regime_mapping, summarize_states
from .visualization import plot_candlestick_with_regimes


def _build_detectors(n_states: int, seed: int, hmm_detector: HMMDetector | None = None) -> dict[str, BaseDetector]:
    return {
        "hmm": hmm_detector or HMMDetector(n_states=n_states, random_state=seed),
        "gmm": GMMDetector(n_states=n_states, random_state=seed),
        "kmeans": KMeansDetector(n_states=n_states, random_state=seed),
    }


def _detector_artifact_path(run_dir: Path, model_name: str) -> Path:
    return run_dir / model_name / f"{model_name}_detector.pkl"


def _load_pretrained_detectors(pretrained_run_dir: Path) -> dict[str, BaseDetector]:
    model_classes: dict[str, type[BaseDetector]] = {
        "hmm": HMMDetector,
        "gmm": GMMDetector,
        "kmeans": KMeansDetector,
    }

    detectors: dict[str, BaseDetector] = {}
    for model_name, detector_cls in model_classes.items():
        model_path = _detector_artifact_path(pretrained_run_dir, model_name)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pretrained detector file not found for '{model_name}': {model_path}. "
                "Run training once to generate detector artifacts."
            )
        detectors[model_name] = detector_cls.load(model_path)

    return detectors


def _apply_labels_and_summary(
    predicted_df: pd.DataFrame,
    state_col: str,
    label_col: str,
    train_idx: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_pred = predicted_df.iloc[:train_idx].copy()
    summary = summarize_states(train_pred, state_col=state_col)
    mapping = infer_regime_mapping(summary, state_col=state_col)

    labeled_df = apply_regime_labels(predicted_df, state_col=state_col, mapping=mapping, label_col=label_col)
    summary["regime_label"] = summary[state_col].map(mapping).fillna("Unknown")
    return labeled_df, summary


def _resolve_hmm_tuning_grid(
    n_states: int,
    seed: int,
    hmm_state_grid: list[int] | None,
    hmm_covariance_grid: list[str] | None,
    hmm_iter_grid: list[int] | None,
    hmm_seed_grid: list[int] | None,
) -> tuple[list[int], list[str], list[int], list[int]]:
    if hmm_state_grid is None:
        state_grid = sorted({s for s in [n_states - 1, n_states, n_states + 1] if s >= 2})
    else:
        state_grid = sorted({int(s) for s in hmm_state_grid if int(s) >= 2})
    if not state_grid:
        raise ValueError("hmm_state_grid must contain at least one value >= 2")

    if hmm_covariance_grid is None:
        covariance_grid = ["diag", "full"]
    else:
        covariance_grid = [str(c).strip().lower() for c in hmm_covariance_grid if str(c).strip()]
    allowed_cov = {"diag", "full", "spherical", "tied"}
    covariance_grid = [c for c in covariance_grid if c in allowed_cov]
    if not covariance_grid:
        raise ValueError("hmm_covariance_grid must contain valid covariance types")

    if hmm_iter_grid is None:
        iter_grid = [200, 400]
    else:
        iter_grid = sorted({int(i) for i in hmm_iter_grid if int(i) >= 50})
    if not iter_grid:
        raise ValueError("hmm_iter_grid must contain at least one value >= 50")

    if hmm_seed_grid is None:
        seed_grid = [seed, seed + 11, seed + 23]
    else:
        seed_grid = sorted({int(s) for s in hmm_seed_grid})
    if not seed_grid:
        raise ValueError("hmm_seed_grid must contain at least one integer")

    return state_grid, covariance_grid, iter_grid, seed_grid


def _tune_hmm_detector(
    features_df: pd.DataFrame,
    train_idx: int,
    run_dir: Path,
    n_states: int,
    seed: int,
    hmm_state_grid: list[int] | None,
    hmm_covariance_grid: list[str] | None,
    hmm_iter_grid: list[int] | None,
    hmm_seed_grid: list[int] | None,
    hmm_tune_train_ratio: float,
) -> tuple[HMMDetector, dict[str, Any]]:
    train_window = features_df.iloc[:train_idx].copy()
    if len(train_window) < 120:
        fallback = HMMDetector(n_states=n_states, random_state=seed)
        return fallback, {
            "hmm_tuning_status": "skipped_too_few_train_samples",
            "hmm_best_n_states": n_states,
            "hmm_best_covariance": "diag",
            "hmm_best_n_iter": 300,
            "hmm_best_seed": seed,
        }

    _, _, tune_train_idx = split_train_test(train_window, train_ratio=hmm_tune_train_ratio)
    x_tune_train = train_window.iloc[:tune_train_idx][FEATURE_COLUMNS].to_numpy()
    x_tune_all = train_window[FEATURE_COLUMNS].to_numpy()

    state_grid, covariance_grid, iter_grid, seed_grid = _resolve_hmm_tuning_grid(
        n_states=n_states,
        seed=seed,
        hmm_state_grid=hmm_state_grid,
        hmm_covariance_grid=hmm_covariance_grid,
        hmm_iter_grid=hmm_iter_grid,
        hmm_seed_grid=hmm_seed_grid,
    )

    candidate_rows: list[dict[str, float | str]] = []
    for n_states_candidate, covariance_type, n_iter_candidate, seed_candidate in product(
        state_grid,
        covariance_grid,
        iter_grid,
        seed_grid,
    ):
        detector = HMMDetector(
            n_states=int(n_states_candidate),
            random_state=int(seed_candidate),
            n_iter=int(n_iter_candidate),
            covariance_type=str(covariance_type),
        )
        try:
            detector.fit(x_tune_train)
            tune_states = detector.predict(x_tune_all)

            state_col = "hmm_tune_state"
            label_col = "hmm_tune_regime"

            predicted_df = train_window.copy()
            predicted_df[state_col] = tune_states
            labeled_df, _ = _apply_labels_and_summary(
                predicted_df=predicted_df,
                state_col=state_col,
                label_col=label_col,
                train_idx=tune_train_idx,
            )

            val_labeled = labeled_df.iloc[tune_train_idx:].copy()
            metrics = evaluate_model(val_labeled, state_col=state_col, label_col=label_col)
            metrics.update(
                {
                    "model": f"hmm_s{n_states_candidate}_{covariance_type}_it{n_iter_candidate}_seed{seed_candidate}",
                    "hmm_n_states": int(n_states_candidate),
                    "hmm_covariance": str(covariance_type),
                    "hmm_n_iter": int(n_iter_candidate),
                    "hmm_seed": int(seed_candidate),
                }
            )
            candidate_rows.append(metrics)
        except Exception:
            continue

    if not candidate_rows:
        fallback = HMMDetector(n_states=n_states, random_state=seed)
        return fallback, {
            "hmm_tuning_status": "fallback_no_valid_candidate",
            "hmm_best_n_states": n_states,
            "hmm_best_covariance": "diag",
            "hmm_best_n_iter": 300,
            "hmm_best_seed": seed,
        }

    tuning_leaderboard = pd.DataFrame(candidate_rows)
    tuning_leaderboard = add_composite_score(tuning_leaderboard)

    hmm_dir = run_dir / "hmm"
    hmm_dir.mkdir(parents=True, exist_ok=True)
    tuning_leaderboard.to_csv(hmm_dir / "hmm_tuning_leaderboard.csv", index=False)

    best = tuning_leaderboard.iloc[0]
    best_detector = HMMDetector(
        n_states=int(best["hmm_n_states"]),
        random_state=int(best["hmm_seed"]),
        n_iter=int(best["hmm_n_iter"]),
        covariance_type=str(best["hmm_covariance"]),
    )

    return best_detector, {
        "hmm_tuning_status": "ok",
        "hmm_tuning_candidates": int(len(tuning_leaderboard)),
        "hmm_best_n_states": int(best["hmm_n_states"]),
        "hmm_best_covariance": str(best["hmm_covariance"]),
        "hmm_best_n_iter": int(best["hmm_n_iter"]),
        "hmm_best_seed": int(best["hmm_seed"]),
        "hmm_best_tuning_score": float(best["composite_score"]),
    }


def _run_single_model(
    model_name: str,
    detector: BaseDetector,
    features_df: pd.DataFrame,
    train_idx: int,
    run_dir: Path,
    fit_model: bool,
    save_detector_artifact: bool,
) -> dict[str, float]:
    state_col = f"{model_name}_state"
    label_col = f"{model_name}_regime"

    x_train = features_df.iloc[:train_idx][FEATURE_COLUMNS].to_numpy()
    x_all = features_df[FEATURE_COLUMNS].to_numpy()

    if fit_model:
        detector.fit(x_train)
    all_states = detector.predict(x_all)

    predicted_df = features_df.copy()
    predicted_df[state_col] = all_states

    predicted_df, summary = _apply_labels_and_summary(
        predicted_df=predicted_df,
        state_col=state_col,
        label_col=label_col,
        train_idx=train_idx,
    )
    test_labeled = predicted_df.iloc[train_idx:].copy()

    model_dir = run_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    if save_detector_artifact:
        detector.save(_detector_artifact_path(run_dir, model_name))

    predicted_df.to_csv(model_dir / f"{model_name}_labels.csv", index=False)

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
    hmm_auto_tune: bool = False,
    hmm_state_grid: list[int] | None = None,
    hmm_covariance_grid: list[str] | None = None,
    hmm_iter_grid: list[int] | None = None,
    hmm_seed_grid: list[int] | None = None,
    hmm_tune_train_ratio: float = 0.8,
    load_models_from: str | Path | None = None,
    save_trained_models: bool = True,
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
    pretrained_mode = load_models_from is not None
    pretrained_path = Path(load_models_from) if load_models_from is not None else None

    if pretrained_mode and hmm_auto_tune:
        raise ValueError("hmm_auto_tune cannot be enabled when load_models_from is used")

    hmm_detector: HMMDetector | None = None
    hmm_tuning_info: dict[str, Any] = {}

    if hmm_auto_tune and not pretrained_mode:
        hmm_detector, hmm_tuning_info = _tune_hmm_detector(
            features_df=features_df,
            train_idx=train_idx,
            run_dir=run_dir,
            n_states=n_states,
            seed=seed,
            hmm_state_grid=hmm_state_grid,
            hmm_covariance_grid=hmm_covariance_grid,
            hmm_iter_grid=hmm_iter_grid,
            hmm_seed_grid=hmm_seed_grid,
            hmm_tune_train_ratio=hmm_tune_train_ratio,
        )

    if pretrained_mode:
        if pretrained_path is None:
            raise ValueError("load_models_from path is required in pretrained mode")
        detectors = _load_pretrained_detectors(pretrained_path)
    else:
        detectors = _build_detectors(n_states=n_states, seed=seed, hmm_detector=hmm_detector)

    for name, detector in detectors.items():
        all_metrics[name] = _run_single_model(
            model_name=name,
            detector=detector,
            features_df=features_df,
            train_idx=train_idx,
            run_dir=run_dir,
            fit_model=not pretrained_mode,
            save_detector_artifact=save_trained_models and not pretrained_mode,
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
            "pretrained_mode": pretrained_mode,
            "load_models_from": str(pretrained_path) if pretrained_path is not None else "",
            "save_trained_models": save_trained_models,
            "hmm_auto_tune": hmm_auto_tune,
            **hmm_tuning_info,
        }
    ]
    pd.DataFrame(summary_rows).to_csv(run_dir / "run_summary.csv", index=False)

    return run_dir
