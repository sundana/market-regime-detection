from __future__ import annotations

from datetime import datetime
from itertools import product
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from .detectors import BaseDetector, GMMDetector, HMMDetector, HMMGMMDetector, KMeansDetector
from .evaluation import add_composite_score, evaluate_model, split_train_test
from .features import FEATURE_COLUMNS, build_feature_table
from .labeling import apply_regime_labels, infer_regime_mapping, summarize_states
from .visualization import plot_candlestick_with_regimes


MODEL_ORDER: tuple[str, ...] = ("hmm", "hmm_gmm", "gmm", "kmeans")
DEFAULT_PRETRAINED_REQUIRED_MODELS: tuple[str, ...] = ("hmm", "gmm", "kmeans")


def _resolve_selected_models(selected_models: list[str] | None) -> list[str]:
    if selected_models is None:
        return list(MODEL_ORDER)

    cleaned = [str(m).strip().lower() for m in selected_models if str(m).strip()]
    if not cleaned:
        raise ValueError("selected_models must contain at least one model name")

    if "all" in cleaned:
        return list(MODEL_ORDER)

    invalid = sorted({m for m in cleaned if m not in MODEL_ORDER})
    if invalid:
        raise ValueError(
            f"Unknown model(s) in selected_models: {', '.join(invalid)}. "
            f"Allowed models: {', '.join(MODEL_ORDER)}"
        )

    selected_set = set(cleaned)
    return [model_name for model_name in MODEL_ORDER if model_name in selected_set]


def _format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0

    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _print_progress_line(prefix: str, current: int, total: int, started_at: float) -> None:
    if total <= 0:
        return

    progress = min(max(current / total, 0.0), 1.0)
    width = 28
    filled = int(width * progress)
    bar = "#" * filled + "-" * (width - filled)

    elapsed = time.perf_counter() - started_at
    rate = current / elapsed if elapsed > 0 else 0.0
    eta = (total - current) / rate if rate > 0 else float("inf")
    eta_text = _format_duration(eta) if eta != float("inf") else "--:--"

    print(
        f"\r{prefix} [{bar}] {current}/{total} ({progress * 100:5.1f}%) "
        f"elapsed {_format_duration(elapsed)} eta {eta_text}",
        end="",
        flush=True,
    )
    if current >= total:
        print("")


def _build_detectors(
    n_states: int,
    seed: int,
    hmm_detector: HMMDetector | None = None,
    hmm_gmm_n_mix: int = 2,
    hmm_gmm_n_iter: int = 300,
    hmm_gmm_covariance_type: str = "diag",
    selected_models: list[str] | None = None,
) -> dict[str, BaseDetector]:
    detectors: dict[str, BaseDetector] = {
        "hmm": hmm_detector or HMMDetector(n_states=n_states, random_state=seed),
        "hmm_gmm": HMMGMMDetector(
            n_states=n_states,
            random_state=seed,
            n_mix=hmm_gmm_n_mix,
            n_iter=hmm_gmm_n_iter,
            covariance_type=hmm_gmm_covariance_type,
        ),
        "gmm": GMMDetector(n_states=n_states, random_state=seed),
        "kmeans": KMeansDetector(n_states=n_states, random_state=seed),
    }

    model_names = _resolve_selected_models(selected_models)
    return {model_name: detectors[model_name] for model_name in model_names}


def _detector_artifact_path(run_dir: Path, model_name: str) -> Path:
    return run_dir / model_name / f"{model_name}_detector.pkl"


def _load_pretrained_detectors(
    pretrained_run_dir: Path,
    selected_models: list[str] | None = None,
) -> dict[str, BaseDetector]:
    model_classes: dict[str, type[BaseDetector]] = {
        "hmm": HMMDetector,
        "hmm_gmm": HMMGMMDetector,
        "gmm": GMMDetector,
        "kmeans": KMeansDetector,
    }

    resolved_models = _resolve_selected_models(selected_models)
    required_models: list[str] = (
        [m for m in DEFAULT_PRETRAINED_REQUIRED_MODELS if m in resolved_models]
        if selected_models is None
        else resolved_models
    )

    detectors: dict[str, BaseDetector] = {}
    for model_name in required_models:
        detector_cls = model_classes[model_name]
        model_path = _detector_artifact_path(pretrained_run_dir, model_name)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pretrained detector file not found for '{model_name}': {model_path}. "
                "Run training once to generate detector artifacts."
            )
        detectors[model_name] = detector_cls.load(model_path)

    if selected_models is None and "hmm_gmm" in resolved_models:
        model_path = _detector_artifact_path(pretrained_run_dir, "hmm_gmm")
        if model_path.exists():
            detectors["hmm_gmm"] = HMMGMMDetector.load(model_path)

    return detectors


def _extract_convergence_info(model_name: str, detector: BaseDetector) -> dict[str, Any]:
    prefix = model_name.lower()
    info: dict[str, Any] = {
        f"{prefix}_convergence_status": "unknown",
        f"{prefix}_convergence_warning": "",
        f"{prefix}_converged": None,
        f"{prefix}_n_iter": None,
        f"{prefix}_max_iter": None,
        f"{prefix}_hit_max_iter": None,
    }

    model = getattr(detector, "model", None)
    if model is None:
        info[f"{prefix}_convergence_status"] = "warning"
        info[f"{prefix}_convergence_warning"] = "Model object not available"
        return info

    warning_msg = ""

    if model_name in {"hmm", "hmm_gmm"}:
        monitor = getattr(model, "monitor_", None)
        converged = bool(getattr(monitor, "converged", False))
        n_iter = int(getattr(monitor, "iter", -1))
        max_iter = int(getattr(model, "n_iter", -1))
        hit_max_iter = bool(max_iter > 0 and n_iter >= max_iter)
        if not converged:
            warning_msg = f"{model_name.upper()} did not report convergence"
        elif hit_max_iter:
            warning_msg = f"{model_name.upper()} reached maximum iterations"
    elif model_name == "gmm":
        converged = bool(getattr(model, "converged_", False))
        n_iter = int(getattr(model, "n_iter_", -1))
        max_iter = int(getattr(model, "max_iter", -1))
        hit_max_iter = bool(max_iter > 0 and n_iter >= max_iter)
        if not converged:
            warning_msg = "GMM did not report convergence"
        elif hit_max_iter:
            warning_msg = "GMM reached maximum iterations"
    elif model_name == "kmeans":
        n_iter = int(getattr(model, "n_iter_", -1))
        max_iter = int(getattr(model, "max_iter", -1))
        hit_max_iter = bool(max_iter > 0 and n_iter >= max_iter)
        converged = bool(not hit_max_iter and n_iter >= 0)
        if hit_max_iter:
            warning_msg = "KMeans reached maximum iterations"
    else:
        return info

    info[f"{prefix}_converged"] = converged
    info[f"{prefix}_n_iter"] = n_iter
    info[f"{prefix}_max_iter"] = max_iter
    info[f"{prefix}_hit_max_iter"] = hit_max_iter
    info[f"{prefix}_convergence_warning"] = warning_msg
    info[f"{prefix}_convergence_status"] = "warning" if warning_msg else "ok"
    return info


def _print_convergence_log(model_name: str, convergence_info: dict[str, Any]) -> None:
    prefix = model_name.lower()
    converged = convergence_info.get(f"{prefix}_converged")
    n_iter = convergence_info.get(f"{prefix}_n_iter")
    max_iter = convergence_info.get(f"{prefix}_max_iter")
    status = convergence_info.get(f"{prefix}_convergence_status", "unknown")
    warning_msg = str(convergence_info.get(f"{prefix}_convergence_warning", "") or "")

    print(
        f"[Stage 4/4] -> {model_name.upper()}: convergence "
        f"status={status} converged={converged} iter={n_iter}/{max_iter}"
    )
    if warning_msg:
        print(f"[Warning] {model_name.upper()} convergence: {warning_msg}")


def _extract_training_diagnostics(
    model_name: str,
    detector: BaseDetector,
    x_train: np.ndarray,
) -> dict[str, Any]:
    prefix = model_name.lower()
    info: dict[str, Any] = {
        f"{prefix}_train_log_likelihood_total": None,
        f"{prefix}_train_log_likelihood_avg": None,
        f"{prefix}_train_log_likelihood_last_iter": None,
        f"{prefix}_train_log_likelihood_delta": None,
        f"{prefix}_train_log_likelihood_history": "",
        f"{prefix}_train_objective_name": "",
        f"{prefix}_train_objective_value": None,
    }

    model = getattr(detector, "model", None)
    if model is None or x_train.size == 0:
        return info

    x_scaled = detector.scaler.transform(x_train)

    if model_name in {"hmm", "hmm_gmm"}:
        total_ll = float(model.score(x_scaled))
        history_raw = list(getattr(getattr(model, "monitor_", None), "history", []))
        history = [float(v) for v in history_raw]

        info[f"{prefix}_train_log_likelihood_total"] = total_ll
        info[f"{prefix}_train_log_likelihood_avg"] = total_ll / float(len(x_train))
        info[f"{prefix}_train_objective_name"] = "log_likelihood"
        info[f"{prefix}_train_objective_value"] = total_ll

        if history:
            info[f"{prefix}_train_log_likelihood_last_iter"] = history[-1]
            info[f"{prefix}_train_log_likelihood_history"] = "|".join(f"{v:.6f}" for v in history)
            if len(history) > 1:
                info[f"{prefix}_train_log_likelihood_delta"] = history[-1] - history[0]
    elif model_name == "gmm":
        avg_ll = float(model.score(x_scaled))
        total_ll = avg_ll * float(len(x_train))
        lower_bound = float(getattr(model, "lower_bound_", np.nan))

        info[f"{prefix}_train_log_likelihood_total"] = total_ll
        info[f"{prefix}_train_log_likelihood_avg"] = avg_ll
        info[f"{prefix}_train_log_likelihood_last_iter"] = lower_bound
        info[f"{prefix}_train_objective_name"] = "lower_bound"
        info[f"{prefix}_train_objective_value"] = lower_bound
    elif model_name == "kmeans":
        inertia = float(getattr(model, "inertia_", np.nan))
        info[f"{prefix}_train_objective_name"] = "inertia"
        info[f"{prefix}_train_objective_value"] = inertia

    return info


def _print_training_diagnostics_log(model_name: str, training_info: dict[str, Any]) -> None:
    prefix = model_name.lower()
    total_ll = training_info.get(f"{prefix}_train_log_likelihood_total")
    avg_ll = training_info.get(f"{prefix}_train_log_likelihood_avg")
    last_iter_ll = training_info.get(f"{prefix}_train_log_likelihood_last_iter")
    delta_ll = training_info.get(f"{prefix}_train_log_likelihood_delta")
    objective_name = str(training_info.get(f"{prefix}_train_objective_name", "") or "")
    objective_value = training_info.get(f"{prefix}_train_objective_value")

    if total_ll is not None:
        msg = (
            f"[Stage 4/4] -> {model_name.upper()}: log_likelihood "
            f"total={float(total_ll):.4f} avg={float(avg_ll):.6f}"
        )
        if last_iter_ll is not None:
            msg += f" last_iter={float(last_iter_ll):.6f}"
        if delta_ll is not None:
            msg += f" delta={float(delta_ll):.6f}"
        print(msg)
    elif objective_name:
        print(
            f"[Stage 4/4] -> {model_name.upper()}: training objective "
            f"{objective_name}={objective_value}"
        )


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

    total_candidates = len(state_grid) * len(covariance_grid) * len(iter_grid) * len(seed_grid)
    print(f"[HMM Tuning] Evaluating {total_candidates} candidate configurations...")
    tune_started = time.perf_counter()

    candidate_rows: list[dict[str, float | str]] = []
    for idx, (n_states_candidate, covariance_type, n_iter_candidate, seed_candidate) in enumerate(
        product(
        state_grid,
        covariance_grid,
        iter_grid,
        seed_grid,
        ),
        start=1,
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
            pass

        _print_progress_line("[HMM Tuning]", idx, total_candidates, tune_started)

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
    generate_chart: bool,
    chart_include_train: bool,
    test_rolling_window: int,
    test_prediction_step: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model_started = time.perf_counter()
    print(f"[Stage 4/4] -> {model_name.upper()}: preparing inputs...")

    state_col = f"{model_name}_state"
    label_col = f"{model_name}_regime"

    x_train = features_df.iloc[:train_idx][FEATURE_COLUMNS].to_numpy()

    if fit_model:
        print(f"[Stage 4/4] -> {model_name.upper()}: training model...")
        detector.fit(x_train)
    else:
        print(f"[Stage 4/4] -> {model_name.upper()}: using pre-trained model (skip training)...")

    convergence_info = _extract_convergence_info(model_name=model_name, detector=detector)
    _print_convergence_log(model_name=model_name, convergence_info=convergence_info)
    training_info = _extract_training_diagnostics(model_name=model_name, detector=detector, x_train=x_train)
    _print_training_diagnostics_log(model_name=model_name, training_info=training_info)

    uses_rolling_context = model_name in {"hmm", "hmm_gmm"}
    if uses_rolling_context:
        print(
            f"[Stage 4/4] -> {model_name.upper()}: predicting states "
            f"(walk-forward, window={test_rolling_window}, step=1 on test)..."
        )
    else:
        print(
            f"[Stage 4/4] -> {model_name.upper()}: predicting states "
            "(point-wise; rolling-window has no effect for this detector)..."
        )

    all_states = np.empty(len(features_df), dtype=int)
    all_states[:train_idx] = detector.predict(x_train)

    rolling_window_start: list[pd.Timestamp | None] = [None] * len(features_df)
    test_total = len(features_df) - train_idx
    rolling_started = time.perf_counter()

    if test_total > 0:
        if uses_rolling_context:
            for offset, row_idx in enumerate(range(train_idx, len(features_df)), start=1):
                window_end_idx = row_idx + 1
                window_start_idx = max(0, window_end_idx - test_rolling_window)
                x_window = features_df.iloc[window_start_idx:window_end_idx][FEATURE_COLUMNS].to_numpy()

                # Walk-forward inference: take only the last state at time t.
                rolling_states = detector.predict(x_window)
                all_states[row_idx] = int(rolling_states[-1])
                rolling_window_start[row_idx] = pd.Timestamp(features_df.iloc[window_start_idx]["timestamp"])

                _print_progress_line(
                    f"[Stage 4/4] -> {model_name.upper()}: walk-forward predict",
                    offset,
                    test_total,
                    rolling_started,
                )
        else:
            x_test = features_df.iloc[train_idx:][FEATURE_COLUMNS].to_numpy()
            all_states[train_idx:] = detector.predict(x_test)
            for offset, row_idx in enumerate(range(train_idx, len(features_df)), start=1):
                _print_progress_line(
                    f"[Stage 4/4] -> {model_name.upper()}: point-wise predict",
                    offset,
                    test_total,
                    rolling_started,
                )

    predicted_df = features_df.copy()
    predicted_df[state_col] = all_states
    predicted_df["rolling_window_start"] = rolling_window_start
    predicted_df["rolling_window_size"] = float(test_rolling_window if uses_rolling_context else 1)
    predicted_df["inference_method"] = "walk_forward" if uses_rolling_context else "pointwise"

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
        print(f"[Stage 4/4] -> {model_name.upper()}: saving detector artifact...")
        detector.save(_detector_artifact_path(run_dir, model_name))

    print(f"[Stage 4/4] -> {model_name.upper()}: writing labels and state summary...")
    predicted_df.to_csv(model_dir / f"{model_name}_labels.csv", index=False)

    summary.to_csv(model_dir / f"{model_name}_state_summary.csv", index=False)

    metrics = evaluate_model(test_labeled, state_col=state_col, label_col=label_col)

    if generate_chart:
        print(f"[Stage 4/4] -> {model_name.upper()}: rendering candlestick HTML chart...")
        chart_df = predicted_df if chart_include_train else test_labeled
        split_ts: pd.Timestamp | None = None
        if chart_include_train and 0 < train_idx < len(predicted_df):
            split_ts = pd.Timestamp(predicted_df.iloc[train_idx]["timestamp"])

        inference_note = (
            "Out-of-sample states predicted with walk-forward inference "
            f"({test_rolling_window} bars context, step=1, last-state-only at each t)"
            if uses_rolling_context
            else "Out-of-sample states predicted point-wise (non-sequential detector; rolling-window has no effect)."
        )
        chart_path = plot_candlestick_with_regimes(
            chart_df,
            label_col=label_col,
            output_path=model_dir / f"{model_name}_candlestick_regime.html",
            title=f"Regime Labeling on Candlestick - {model_name.upper()}",
            inference_note=inference_note,
            show_rolling_points=uses_rolling_context,
            train_test_split_ts=split_ts,
        )
        metrics["chart_path"] = str(chart_path)
    else:
        metrics["chart_path"] = ""

    model_elapsed = time.perf_counter() - model_started
    print(f"[Stage 4/4] -> {model_name.upper()}: done in {_format_duration(model_elapsed)}")

    return metrics, convergence_info, training_info


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
    hmm_gmm_n_mix: int = 2,
    hmm_gmm_n_iter: int = 300,
    hmm_gmm_covariance_type: str = "diag",
    selected_models: list[str] | None = None,
    load_models_from: str | Path | None = None,
    save_trained_models: bool = True,
    generate_charts: bool = True,
    chart_include_train: bool = False,
    test_rolling_window: int = 120,
    test_prediction_step: int = 1,
) -> Path:
    selected_models = _resolve_selected_models(selected_models)

    if test_rolling_window < 2:
        raise ValueError("test_rolling_window must be >= 2")
    if test_prediction_step != 1:
        raise ValueError("Walk-forward inference requires test_prediction_step=1")
    if hmm_gmm_n_mix < 1:
        raise ValueError("hmm_gmm_n_mix must be >= 1")
    if hmm_gmm_n_iter < 1:
        raise ValueError("hmm_gmm_n_iter must be >= 1")
    hmm_gmm_covariance_type = str(hmm_gmm_covariance_type).strip().lower()
    allowed_hmm_gmm_covariance = {"diag", "full", "spherical", "tied"}
    if hmm_gmm_covariance_type not in allowed_hmm_gmm_covariance:
        raise ValueError(
            "hmm_gmm_covariance_type must be one of: diag, full, spherical, tied"
        )

    experiment_started = time.perf_counter()
    print("[Stage 1/4] Building feature table...")
    feature_started = time.perf_counter()
    features_df = build_feature_table(
        data_root=data_root,
        pair=pair,
        timeframe=timeframe,
        return_lag=1,
        max_bars=max_bars,
    )
    feature_elapsed = time.perf_counter() - feature_started
    print(f"[Stage 1/4] Done in {_format_duration(feature_elapsed)} ({len(features_df):,} rows)")

    print("[Stage 2/4] Splitting train/test...")

    _, _, train_idx = split_train_test(features_df, train_ratio=train_ratio)
    print(
        f"[Stage 2/4] Done. train={train_idx:,} test={len(features_df) - train_idx:,} "
        f"(ratio={train_ratio:.2f})"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / "regime_detection" / f"{pair}_{timeframe}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    features_df.to_csv(run_dir / "feature_table.csv", index=False)

    all_metrics: dict[str, dict[str, Any]] = {}
    convergence_summary: dict[str, Any] = {}
    training_diagnostics_summary: dict[str, Any] = {}
    pretrained_mode = load_models_from is not None
    pretrained_path = Path(load_models_from) if load_models_from is not None else None

    if pretrained_mode and hmm_auto_tune:
        raise ValueError("hmm_auto_tune cannot be enabled when load_models_from is used")
    if hmm_auto_tune and "hmm" not in selected_models:
        raise ValueError("hmm_auto_tune requires 'hmm' to be included in selected_models")

    hmm_detector: HMMDetector | None = None
    hmm_tuning_info: dict[str, Any] = {}

    print("[Stage 3/4] Preparing detectors...")
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
        print(f"[Stage 3/4] Loading pretrained detectors from: {pretrained_path}")
        detectors = _load_pretrained_detectors(pretrained_path, selected_models=selected_models)
    else:
        detectors = _build_detectors(
            n_states=n_states,
            seed=seed,
            hmm_detector=hmm_detector,
            hmm_gmm_n_mix=hmm_gmm_n_mix,
            hmm_gmm_n_iter=hmm_gmm_n_iter,
            hmm_gmm_covariance_type=hmm_gmm_covariance_type,
            selected_models=selected_models,
        )
    print("[Stage 3/4] Done.")

    print("[Stage 4/4] Running model training/evaluation...")
    model_started = time.perf_counter()
    model_total = len(detectors)

    for idx, (name, detector) in enumerate(detectors.items(), start=1):
        print(f"[Stage 4/4] Starting model {idx}/{model_total}: {name.upper()}")
        model_metrics, model_convergence, model_training_info = _run_single_model(
            model_name=name,
            detector=detector,
            features_df=features_df,
            train_idx=train_idx,
            run_dir=run_dir,
            fit_model=not pretrained_mode,
            save_detector_artifact=save_trained_models and not pretrained_mode,
            generate_chart=generate_charts,
            chart_include_train=chart_include_train,
            test_rolling_window=test_rolling_window,
            test_prediction_step=test_prediction_step,
        )
        all_metrics[name] = model_metrics
        convergence_summary.update(model_convergence)
        training_diagnostics_summary.update(model_training_info)
        _print_progress_line("[Stage 4/4]", idx, model_total, model_started)

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
            "generate_charts": generate_charts,
            "chart_include_train": chart_include_train,
            "test_prediction_mode": "walk_forward",
            "test_rolling_window": test_rolling_window,
            "test_prediction_step": test_prediction_step,
            "selected_models": "|".join(selected_models),
            "hmm_auto_tune": hmm_auto_tune,
            "hmm_gmm_n_mix": hmm_gmm_n_mix,
            "hmm_gmm_n_iter": hmm_gmm_n_iter,
            "hmm_gmm_covariance_type": hmm_gmm_covariance_type,
            **hmm_tuning_info,
            **convergence_summary,
            **training_diagnostics_summary,
        }
    ]
    pd.DataFrame(summary_rows).to_csv(run_dir / "run_summary.csv", index=False)

    total_elapsed = time.perf_counter() - experiment_started
    print(f"[Complete] Finished in {_format_duration(total_elapsed)}")

    return run_dir
