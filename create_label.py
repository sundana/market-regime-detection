from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from models.regime_detection.visualization import plot_candlestick_with_regimes

VALID_REGIMES = {"Bullish", "Bearish", "Sideway"}
REGIME_ALIASES = {
	"bullish": "Bullish",
	"bearish": "Bearish",
	"bearsih": "Bearish",
	"sideway": "Sideway",
	"sideways": "Sideway",
}


def _normalize_regime(value: str) -> str:
	normalized = REGIME_ALIASES.get(str(value).strip().lower())
	if normalized is None:
		raise ValueError(f"Unsupported regime label: {value!r}")
	return normalized


def _load_csv(path: Path, timestamp_col: str) -> pd.DataFrame:
	df = pd.read_csv(path, skipinitialspace=True)
	df.columns = [column.strip() for column in df.columns]

	if timestamp_col not in df.columns:
		raise ValueError(f"Missing timestamp column '{timestamp_col}' in {path}")

	df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="raise")
	for column in df.columns:
		if column == timestamp_col:
			continue
		if is_object_dtype(df[column]) or is_string_dtype(df[column]):
			df[column] = df[column].astype(str).str.strip()
	return df


def build_segmented_labels(
	target_df: pd.DataFrame,
	source_df: pd.DataFrame,
	*,
	timestamp_col: str = "timestamp",
	source_col: str = "regime",
	target_col: str = "hmm_regime",
) -> pd.DataFrame:
	if source_col not in source_df.columns:
		raise ValueError(f"Missing source column '{source_col}'")

	labeled = target_df.copy()
	labeled[timestamp_col] = pd.to_datetime(labeled[timestamp_col], errors="raise")
	labeled = labeled.sort_values(timestamp_col).reset_index(drop=True)

	source = source_df[[timestamp_col, source_col]].copy()
	source = source.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
	source = source.drop_duplicates(subset=[timestamp_col], keep="last").reset_index(drop=True)
	source[source_col] = source[source_col].map(_normalize_regime)

	if target_col not in labeled.columns:
		labeled[target_col] = pd.NA

	if source.empty:
		return labeled

	source_timestamps = source[timestamp_col].to_list()
	source_regimes = source[source_col].to_list()

	for index, start_ts in enumerate(source_timestamps):
		regime = source_regimes[index]
		end_ts = source_timestamps[index + 1] if index + 1 < len(source_timestamps) else None

		if end_ts is None:
			mask = labeled[timestamp_col] >= start_ts
		else:
			mask = (labeled[timestamp_col] >= start_ts) & (labeled[timestamp_col] < end_ts)

		if mask.any():
			labeled.loc[mask, target_col] = regime

	return labeled


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Fill ground_truth hmm_regime values from test_label segment timestamps."
	)
	parser.add_argument("--test-labels", type=Path, default=Path("test_label.csv"), help="Path to test_label.csv.")
	parser.add_argument(
		"--ground-truth",
		type=Path,
		default=Path("ground_truth.csv"),
		help="Path to the target ground_truth.csv file.",
	)
	parser.add_argument(
		"--reference-labels",
		type=Path,
		default=None,
		help="Optional hmm_labels.csv used to validate label timestamps before writing.",
	)
	parser.add_argument(
		"--timestamp-column",
		type=str,
		default="timestamp",
		help="Name of the timestamp column.",
	)
	parser.add_argument(
		"--source-column",
		type=str,
		default="regime",
		help="Name of the regime column in test_label.csv.",
	)
	parser.add_argument(
		"--target-column",
		type=str,
		default="hmm_regime",
		help="Name of the column to overwrite in ground_truth.csv.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Optional output path. Defaults to overwriting ground_truth.csv.",
	)
	parser.add_argument(
		"--html-output",
		type=Path,
		default=None,
		help="Optional Plotly HTML output path for the updated ground_truth chart.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if not args.test_labels.exists():
		raise FileNotFoundError(f"Test label file not found: {args.test_labels}")
	if not args.ground_truth.exists():
		raise FileNotFoundError(f"Ground truth file not found: {args.ground_truth}")
	if args.reference_labels is not None and not args.reference_labels.exists():
		raise FileNotFoundError(f"Reference label file not found: {args.reference_labels}")

	test_labels = _load_csv(args.test_labels, args.timestamp_column)
	ground_truth = _load_csv(args.ground_truth, args.timestamp_column)

	if args.reference_labels is not None:
		reference_labels = _load_csv(args.reference_labels, args.timestamp_column)
		missing_timestamps = test_labels.loc[
			~test_labels[args.timestamp_column].isin(reference_labels[args.timestamp_column]),
			args.timestamp_column,
		]
		if not missing_timestamps.empty:
			missing_list = ", ".join(ts.strftime("%Y-%m-%d %H:%M:%S") for ts in missing_timestamps)
			raise ValueError(
				"Some test_label timestamps do not exist in reference_labels: "
				f"{missing_list}"
			)

	updated = build_segmented_labels(
		ground_truth,
		test_labels,
		timestamp_col=args.timestamp_column,
		source_col=args.source_column,
		target_col=args.target_column,
	)

	unknown_regimes = sorted(set(updated[args.target_column].dropna().astype(str)) - VALID_REGIMES)
	if unknown_regimes:
		raise ValueError(f"Unexpected regime values after normalization: {unknown_regimes}")

	output_path = args.output or args.ground_truth
	updated.to_csv(output_path, index=False)

	if args.html_output is not None:
		plot_candlestick_with_regimes(
			updated,
			label_col=args.target_column,
			output_path=args.html_output,
			title=f"Ground Truth Regimes - {args.ground_truth.stem}",
			inference_note="Generated from test_label.csv segment labels",
			show_rolling_points=False,
		)

	print(
		f"Updated '{args.target_column}' in {output_path} using {len(test_labels)} label rows from {args.test_labels}."
	)
	if args.html_output is not None:
		print(f"Plotly chart saved to {args.html_output}")


if __name__ == "__main__":
	main()
