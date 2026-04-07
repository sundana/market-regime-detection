# market-regime-detection

End-to-end benchmarking workflow for market regime detection on candlestick data.

The project now includes:
- Data processing from tick to OHLCV
- Feature engineering for regime models
- Model comparison for HMM, GMM, and K-Means
- Evaluation focused on regime stability and economic validity
- Interactive candlestick visualization with clear regime background labeling

## Workflow

1. Load parquet tick data from `data/<pair>/`.
2. Preprocess and aggregate to OHLCV (`1h` by default).
3. Build regime features:
	- `return_1`
	- `volatility_24`
	- `volatility_72`
	- `range_ratio`
	- `body_ratio`
	- `volume_change`
4. Train and predict with 3 detectors:
	- HMM (hmmlearn)
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
- `models/regime_detection/detectors.py` - HMM, GMM, K-Means wrappers
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

Fast smoke test (use latest bars only):

```bash
python run_regime_experiment.py --pair xauusd --timeframe 1h --max-bars 3000 --states 3 --train-ratio 0.7
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

- When `--max-bars` is used, feature loading uses the latest yearly parquet file to speed up experiments.
- Output charts are interactive HTML (Plotly), optimized for quick inspection of regime labeling clarity on candles.
