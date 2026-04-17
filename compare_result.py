from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VALID_REGIMES = {"Bullish", "Bearish", "Sideway"}
REGIME_ALIASES = {
	"bullish": "Bullish",
	"bearish": "Bearish",
	"bearsih": "Bearish",
	"sideway": "Sideway",
	"sideways": "Sideway",
}


def _normalize_regime(value: object) -> str:
	normalized = REGIME_ALIASES.get(str(value).strip().lower())
	if normalized is None:
		raise ValueError(f"Unsupported regime label: {value!r}")
	return normalized


def _load_labels(path: Path, timestamp_col: str, label_col: str) -> pd.DataFrame:
	df = pd.read_csv(path, skipinitialspace=True)
	df.columns = [column.strip() for column in df.columns]

	if timestamp_col not in df.columns:
		raise ValueError(f"Missing timestamp column '{timestamp_col}' in {path}")
	if label_col not in df.columns:
		raise ValueError(f"Missing label column '{label_col}' in {path}")

	df = df[[timestamp_col, label_col]].copy()
	df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="raise")
	df[label_col] = df[label_col].map(_normalize_regime)
	df = df.dropna(subset=[timestamp_col, label_col])
	df = df.drop_duplicates(subset=[timestamp_col], keep="last")
	return df.sort_values(timestamp_col).reset_index(drop=True)


def compare_labels(
	ground_truth: pd.DataFrame,
	predictions: pd.DataFrame,
	*,
	timestamp_col: str = "timestamp",
	truth_col: str = "hmm_regime",
	pred_col: str = "hmm_regime",
) -> tuple[pd.DataFrame, pd.DataFrame]:
	merged = ground_truth.merge(
		predictions,
		on=timestamp_col,
		how="inner",
		suffixes=("_truth", "_pred"),
	)

	if merged.empty:
		raise ValueError("No matching timestamps were found between ground truth and predictions.")

	truth_merged_col = f"{truth_col}_truth"
	pred_merged_col = f"{pred_col}_pred"
	if truth_merged_col not in merged.columns:
		raise ValueError(f"Expected merged truth column '{truth_merged_col}' not found")
	if pred_merged_col not in merged.columns:
		raise ValueError(f"Expected merged prediction column '{pred_merged_col}' not found")

	merged["is_correct"] = merged[truth_merged_col] == merged[pred_merged_col]
	errors = merged.loc[~merged["is_correct"], [timestamp_col, truth_merged_col, pred_merged_col]].copy()
	errors = errors.rename(columns={truth_merged_col: "ground_truth", pred_merged_col: "predicted"})
	return merged, errors


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Compare ground truth regimes against HMM prediction regimes by timestamp."
	)
	parser.add_argument(
		"--ground-truth",
		type=Path,
		default=Path("ground_truth.csv"),
		help="Path to ground_truth.csv.",
	)
	parser.add_argument(
		"--hmm-labels",
		"--predictions",
		type=Path,
		default=None,
		help="Path to a specific hmm_labels.csv file. If omitted, the latest results/**/hmm/hmm_labels.csv is used when available.",
	)
	parser.add_argument(
		"--timestamp-column",
		type=str,
		default="timestamp",
		help="Timestamp column name used for alignment.",
	)
	parser.add_argument(
		"--label-column",
		type=str,
		default="hmm_regime",
		help="Label column name to compare.",
	)
	parser.add_argument(
		"--top-errors",
		type=int,
		default=20,
		help="Maximum number of mismatch rows to print.",
	)
	return parser.parse_args()


def _find_latest_prediction_file() -> Path:
	candidates = sorted(
		Path("results").glob("regime_detection/*/hmm/hmm_labels.csv"),
		key=lambda path: path.stat().st_mtime,
		reverse=True,
	)
	if not candidates:
		raise FileNotFoundError(
			"No prediction file provided and no results/**/hmm/hmm_labels.csv file was found."
		)
	return candidates[0]


def main() -> None:
	args = parse_args()

	if not args.ground_truth.exists():
		raise FileNotFoundError(f"Ground truth file not found: {args.ground_truth}")

	prediction_path = args.hmm_labels or _find_latest_prediction_file()
	if not prediction_path.exists():
		raise FileNotFoundError(f"Prediction file not found: {prediction_path}")

	ground_truth = _load_labels(args.ground_truth, args.timestamp_column, args.label_column)
	predictions = _load_labels(prediction_path, args.timestamp_column, args.label_column)

	merged, errors = compare_labels(
		ground_truth,
		predictions,
		timestamp_col=args.timestamp_column,
		truth_col=args.label_column,
		pred_col=args.label_column,
	)

	total = len(merged)
	correct = int(merged["is_correct"].sum())
	accuracy = correct / total if total else 0.0
	error_count = total - correct
	error_rate = error_count / total if total else 0.0

	missing_truth = len(ground_truth) - total
	missing_pred = len(predictions) - total

	print(f"Ground truth file : {args.ground_truth}")
	print(f"Prediction file   : {prediction_path}")
	print(f"Matched timestamps: {total}")
	print(f"Unmatched ground truth rows: {missing_truth}")
	print(f"Unmatched prediction rows  : {missing_pred}")
	print(f"Accuracy          : {accuracy:.4%} ({correct}/{total})")
	print(f"Error rate        : {error_rate:.4%} ({error_count}/{total})")

	if errors.empty:
		print("No mismatches found.")
		return

	print("")
	print(f"Mismatches (showing up to {args.top_errors} rows):")
	preview = errors.head(max(args.top_errors, 0))
	for _, row in preview.iterrows():
		print(
			f"- {row[args.timestamp_column]} | ground_truth={row['ground_truth']} | predicted={row['predicted']}"
		)


if __name__ == "__main__":
	main()
