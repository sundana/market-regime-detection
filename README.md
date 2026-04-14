# market-regime-detection

End-to-end benchmarking workflow for market regime detection on candlestick data.

The project now includes:
- Data processing from tick to OHLCV
- Feature engineering for regime models
- Model comparison for HMM, HMM-GMM, GMM, and K-Means
- Evaluation focused on regime stability and economic validity
- Interactive candlestick visualization with clear regime background labeling

## Workflow

1. Load parquet tick data from `data/<pair>/`.
2. Preprocess and aggregate to OHLCV (`1h` by default).
3. Build regime features:
	- `return_1`
	- `rolling_dev_return_14`
	- `volatility_24`
	- `volatility_72`
	- `range_ratio`
	- `body_ratio`
	- `volume_change`
4. Train and predict with 4 detectors:
	- HMM (hmmlearn)
	- HMM-GMM (hmmlearn GMMHMM)
	- GMM (scikit-learn)
	- K-Means (scikit-learn)
5. Map numeric states into readable regime labels:
	- `Bullish`
	- `Bearish`
	- `Sideways`
	- `Volatile`
6. Evaluate model quality on test split.
7. Export interactive Plotly candlestick charts with regime background shading.

## New Files

- `run_regime_experiment.py` - One-command experiment runner
- `models/regime_detection/features.py` - Feature pipeline
- `models/regime_detection/detectors.py` - HMM, HMM-GMM, GMM, K-Means wrappers
- `models/regime_detection/labeling.py` - State to regime labeling
- `models/regime_detection/evaluation.py` - Metrics and scoring
- `models/regime_detection/visualization.py` - Plotly candlestick regime chart
- `models/regime_detection/pipeline.py` - Orchestration logic

## Quick Start

### 1) Environment

Create or update environment:

```bash
conda env create -f environment.yaml
```

If environment already exists, make sure this dependency is installed:

```bash
pip install hmmlearn
```

### 2) Run Regime Benchmark

Basic run (xauusd, 1H):

```bash
python run_regime_experiment.py --pair xauusd --timeframe 1h --states 3 --train-ratio 0.7
```

Run only selected models:

```bash
python run_regime_experiment.py --pair xauusd --timeframe 1h --states 3 --models hmm,hmm_gmm
```

Fast smoke test (use latest bars only):

```bash
python run_regime_experiment.py --pair xauusd --timeframe 1h --max-bars 3000 --states 3 --train-ratio 0.7
```

HMM auto-tuning (multi-state, covariance, iteration, and seed search):

```bash
python run_regime_experiment.py --pair xauusd --timeframe 1h --max-bars 3000 --states 3 --train-ratio 0.7 --hmm-auto-tune
```

Custom HMM tuning grid:

```bash
python run_regime_experiment.py --pair xauusd --timeframe 1h --max-bars 3000 --states 3 --train-ratio 0.7 --hmm-auto-tune --hmm-state-grid 2,3,4 --hmm-covariance-grid diag,full --hmm-iter-grid 200,400,600 --hmm-seed-grid 42,53,65 --hmm-tune-train-ratio 0.8
```

Custom HMM-GMM emission complexity:

```bash
python run_regime_experiment.py --pair xauusd --timeframe 1h --max-bars 3000 --states 3 --train-ratio 0.7 --hmm-gmm-mixtures 3 --hmm-gmm-iter 400 --hmm-gmm-covariance full
```

Load pre-trained models (evaluation without retraining):

```bash
python run_regime_experiment.py --pair xauusd --timeframe 1h --max-bars 3000 --train-ratio 0.7 --load-models-from results/regime_detection/xauusd_1h_YYYYMMDD_HHMMSS
```

Optional: do not save detector artifacts during training:

```bash
python run_regime_experiment.py --pair xauusd --timeframe 1h --max-bars 3000 --no-save-models
```

Optional: skip HTML chart rendering for faster runs:

```bash
python run_regime_experiment.py --pair xauusd --timeframe 1h --max-bars 3000 --no-charts
```

Optional: walk-forward inference on test set (hour-by-hour):

```bash
python run_regime_experiment.py --pair xauusd --timeframe 1h --max-bars 3000 --train-ratio 0.7 --rolling-window 120 --rolling-step 1
```

## Output Artifacts

Each run writes to:

`results/regime_detection/<pair>_<timeframe>_<timestamp>/`

Generated files:
- `feature_table.csv`
- `leaderboard.csv`
- `run_summary.csv`
- `<model>/<model>_labels.csv`
- `<model>/<model>_state_summary.csv`
- `<model>/<model>_candlestick_regime.html`
- `<model>/<model>_detector.pkl` (saved trained detector, unless `--no-save-models` is used)
- `hmm/hmm_tuning_leaderboard.csv` (only when `--hmm-auto-tune` is enabled)

## Evaluation Metrics

Primary metrics:
- `avg_regime_duration`
- `transitions_per_100`
- `persistence_ratio`
- `return_separation`
- `volatility_separation`
- `vol_spike_alignment`

Composite model ranking is saved in `leaderboard.csv` via `composite_score`.

## Notes

- When `--max-bars` is used, feature loading uses full available history, then keeps only the latest N feature bars.
- Output charts are interactive HTML (Plotly), optimized for quick inspection of regime labeling clarity on candles.
- HMM auto-tuning uses an internal time-based split inside training data to choose the best HMM configuration before final test evaluation.
- `--load-models-from` expects detector files (`*_detector.pkl`) inside each model subfolder and skips model fitting. `hmm`, `gmm`, and `kmeans` are required; `hmm_gmm` is loaded when available.
- `--models` can be used to run only selected detectors (e.g., `--models hmm,gmm`). Allowed values: `hmm`, `hmm_gmm`, `gmm`, `kmeans`, or `all`.
- Runtime logs now include stage-based progress + ETA for data preparation, HMM tuning, and model training/evaluation.
- Stage 4 now prints per-model substeps so long operations (especially chart rendering) remain visible in terminal output.
- Test inference uses model-aware behavior: HMM uses walk-forward context (`--rolling-window`), while GMM/KMeans are point-wise (non-sequential), so rolling-window does not change their predictions.
- Training now prints convergence diagnostics per model (HMM, GMM, KMeans), with automatic warning lines if convergence is not achieved.
- `run_summary.csv` now stores per-model convergence fields such as converged flag, iteration count, max iteration, and warning message.
