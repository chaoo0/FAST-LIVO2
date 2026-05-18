#!/usr/bin/env python3
"""
Build fixed-length Mamba pose input sequences from cleaned FAST-LIVO2 CSV data.

This script is intentionally limited to offline data preparation. It does not
modify FAST-LIVO2 C++ runtime code, PoseCompensator inference logic, ONNX
backend behavior, or the safety layer.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def require_dependency(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency '{module_name}'. Install it first, for example:\n"
            f"  {install_hint}"
        ) from exc


np = require_dependency("numpy", "python3 -m pip install pandas numpy")
pd = require_dependency("pandas", "python3 -m pip install pandas numpy")


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "Log" / "mamba_pose_train_data_clean.csv"
DEFAULT_OUTPUT = REPO_ROOT / "Log" / "mamba_pose_sequences_T10.npz"
DEFAULT_REPORT = REPO_ROOT / "Log" / "mamba_pose_sequences_T10_report.txt"

BASIC18_FEATURE_NAMES = [
    "pos_x",
    "pos_y",
    "pos_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "rot_w",
    "vel_x",
    "vel_y",
    "vel_z",
    "bias_g_x",
    "bias_g_y",
    "bias_g_z",
    "bias_a_x",
    "bias_a_y",
    "bias_a_z",
    "effective_feature_num",
    "avg_residual",
]

REQUIRED_COLUMNS = ["timestamp"] + BASIC18_FEATURE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build fixed-length sequence inputs for later real Mamba pose "
            "correction model training from cleaned CSV data."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input clean CSV path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output NPZ path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Output TXT report path. Default: {DEFAULT_REPORT}",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=10,
        help="Sequence length T. Default: 10",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding-window stride. Default: 1",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["basic18"],
        default="basic18",
        help="Feature layout mode. Default: basic18",
    )
    parser.add_argument(
        "--max-time-gap",
        type=float,
        default=0.2,
        help="Maximum allowed timestamp gap in seconds for same segment. Default: 0.2",
    )
    parser.add_argument(
        "--allow-time-gap",
        action="store_true",
        help="Disable timestamp-gap segmentation and allow full-table sliding windows.",
    )
    return parser.parse_args()


def ensure_input_exists(input_path: Path) -> None:
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")
    if not input_path.is_file():
        raise SystemExit(f"Input path is not a file: {input_path}")


def load_dataframe(input_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(input_path)
    except Exception as exc:
        raise SystemExit(f"Failed to read CSV '{input_path}': {exc}") from exc


def validate_args(args: argparse.Namespace) -> None:
    if args.seq_len <= 0:
        raise SystemExit("--seq-len must be positive.")
    if args.stride <= 0:
        raise SystemExit("--stride must be positive.")
    if args.max_time_gap <= 0:
        raise SystemExit("--max-time-gap must be positive.")


def validate_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise SystemExit(
            "CSV is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing_columns)
        )


def format_scalar(value) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        if math.isinf(float(value)):
            return "inf" if value > 0 else "-inf"
        return f"{float(value):.6f}"
    return str(value)


def count_windows(length: int, seq_len: int, stride: int) -> int:
    if length < seq_len:
        return 0
    return ((length - seq_len) // stride) + 1


def get_feature_names(feature_mode: str) -> list[str]:
    if feature_mode != "basic18":
        raise SystemExit(f"Unsupported --feature-mode: {feature_mode}")
    return list(BASIC18_FEATURE_NAMES)


def get_source_index_column(df: pd.DataFrame) -> str:
    if "original_index" in df.columns:
        return "original_index"
    if "source_index" in df.columns:
        return "source_index"
    return "clean_csv_row_index"


def prepare_sorted_dataframe(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    prepared = df.copy()
    prepared["clean_csv_row_index"] = np.arange(len(prepared), dtype=np.int64)

    numeric_columns = REQUIRED_COLUMNS
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    invalid_numeric_rows = prepared[REQUIRED_COLUMNS].isna().any(axis=1)
    if invalid_numeric_rows.any():
        count = int(invalid_numeric_rows.sum())
        raise SystemExit(
            "Input clean CSV still contains non-numeric or missing values in required "
            f"fields after cleaning. Invalid row count: {count}"
        )

    sorted_df = prepared.sort_values(
        by=["timestamp", "clean_csv_row_index"], kind="mergesort"
    ).reset_index(drop=True)
    return sorted_df


def compute_original_timestamp_info(df: pd.DataFrame) -> dict[str, float | int | bool]:
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    if timestamps.size <= 1:
        diffs = np.array([], dtype=np.float64)
    else:
        diffs = np.diff(timestamps)

    duplicate_count = int(np.sum(diffs == 0))
    non_monotonic_count = int(np.sum(diffs < 0))
    return {
        "duplicate_timestamp_pairs_in_original_order": duplicate_count,
        "non_monotonic_pairs_in_original_order": non_monotonic_count,
        "original_order_is_strictly_increasing": bool(np.all(diffs > 0)) if diffs.size else True,
    }


def compute_sorted_gap_info(sorted_df: pd.DataFrame) -> dict[str, object]:
    timestamps = sorted_df["timestamp"].to_numpy(dtype=np.float64)
    if timestamps.size <= 1:
        diffs = np.array([], dtype=np.float64)
    else:
        diffs = np.diff(timestamps)

    non_positive_mask = diffs <= 0
    return {
        "sorted_timestamps": timestamps,
        "sorted_diffs": diffs,
        "max_gap": float(diffs.max()) if diffs.size else 0.0,
        "mean_gap": float(diffs.mean()) if diffs.size else 0.0,
        "non_positive_gap_count": int(non_positive_mask.sum()),
        "duplicate_or_reversed_gap_indices": np.where(non_positive_mask)[0],
    }


def build_segments(
    num_rows: int,
    sorted_diffs: np.ndarray,
    max_time_gap: float,
    allow_time_gap: bool,
) -> list[tuple[int, int]]:
    if num_rows == 0:
        return []
    if allow_time_gap:
        return [(0, num_rows)]

    segments: list[tuple[int, int]] = []
    start = 0
    for diff_index, dt in enumerate(sorted_diffs):
        if dt <= 0 or dt > max_time_gap:
            end = diff_index + 1
            segments.append((start, end))
            start = end
    segments.append((start, num_rows))
    return segments


def build_sequence_arrays(
    sorted_df: pd.DataFrame,
    feature_names: list[str],
    source_index_column: str,
    segments: list[tuple[int, int]],
    seq_len: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_array = sorted_df[feature_names].to_numpy(dtype=np.float32)
    timestamp_array = sorted_df["timestamp"].to_numpy(dtype=np.float64)
    source_index_array = pd.to_numeric(
        sorted_df[source_index_column], errors="coerce"
    ).to_numpy(dtype=np.int64)

    x_sequences: list[np.ndarray] = []
    timestamp_sequences: list[np.ndarray] = []
    source_sequences: list[np.ndarray] = []

    for start, end in segments:
        segment_length = end - start
        if segment_length < seq_len:
            continue
        for window_start in range(start, end - seq_len + 1, stride):
            window_end = window_start + seq_len
            x_sequences.append(feature_array[window_start:window_end])
            timestamp_sequences.append(timestamp_array[window_start:window_end])
            source_sequences.append(source_index_array[window_start:window_end])

    if x_sequences:
        x = np.stack(x_sequences, axis=0).astype(np.float32, copy=False)
        timestamps = np.stack(timestamp_sequences, axis=0).astype(np.float64, copy=False)
        source_indices = np.stack(source_sequences, axis=0).astype(np.int64, copy=False)
    else:
        x = np.empty((0, seq_len, len(feature_names)), dtype=np.float32)
        timestamps = np.empty((0, seq_len), dtype=np.float64)
        source_indices = np.empty((0, seq_len), dtype=np.int64)

    return x, timestamps, source_indices


def build_report_lines(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    total_rows: int,
    seq_len: int,
    stride: int,
    feature_names: list[str],
    source_index_column: str,
    max_time_gap: float,
    allow_time_gap: bool,
    total_candidate_windows: int,
    kept_windows: int,
    segments: list[tuple[int, int]],
    sorted_gap_info: dict[str, object],
    original_timestamp_info: dict[str, float | int | bool],
    x: np.ndarray,
    timestamps: np.ndarray,
    source_indices: np.ndarray,
) -> list[str]:
    skipped_by_time_gap = total_candidate_windows - kept_windows
    segment_lengths = np.array([end - start for start, end in segments], dtype=np.int64)

    if segment_lengths.size:
        segment_min = int(segment_lengths.min())
        segment_max = int(segment_lengths.max())
        segment_mean = float(segment_lengths.mean())
    else:
        segment_min = 0
        segment_max = 0
        segment_mean = 0.0

    lines = [
        "Mamba Pose Sequence Build Report",
        "=" * 32,
        "",
        "Input / Output",
        f"- input_csv: {input_path}",
        f"- output_npz: {output_path}",
        f"- output_report: {report_path}",
        "",
        "Build Parameters",
        f"- total_rows: {total_rows}",
        f"- seq_len: {seq_len}",
        f"- stride: {stride}",
        f"- feature_mode: basic18",
        f"- feature_dim: {len(feature_names)}",
        f"- source_index_column: {source_index_column}",
        f"- max_time_gap: {format_scalar(max_time_gap)}",
        f"- allow_time_gap: {'true' if allow_time_gap else 'false'}",
        "",
        "Timestamp Checks",
        (
            "- original_order_is_strictly_increasing: "
            f"{'true' if original_timestamp_info['original_order_is_strictly_increasing'] else 'false'}"
        ),
        (
            "- non_monotonic_pairs_in_original_order: "
            f"{original_timestamp_info['non_monotonic_pairs_in_original_order']}"
        ),
        (
            "- duplicate_timestamp_pairs_in_original_order: "
            f"{original_timestamp_info['duplicate_timestamp_pairs_in_original_order']}"
        ),
        (
            "- max_timestamp_gap_after_sort: "
            f"{format_scalar(sorted_gap_info['max_gap'])}"
        ),
        (
            "- mean_timestamp_gap_after_sort: "
            f"{format_scalar(sorted_gap_info['mean_gap'])}"
        ),
        (
            "- non_positive_gap_count_after_sort: "
            f"{sorted_gap_info['non_positive_gap_count']}"
        ),
        "",
        "Window Statistics",
        f"- total_candidate_windows: {total_candidate_windows}",
        f"- skipped_by_time_gap: {skipped_by_time_gap}",
        f"- kept_windows: {kept_windows}",
        f"- segment_count: {len(segments)}",
        (
            "- segment_length_stats: "
            f"min={segment_min}, max={segment_max}, mean={format_scalar(segment_mean)}"
        ),
        "",
        "Saved Array Shapes",
        f"- X shape: {list(x.shape)}",
        f"- timestamps shape: {list(timestamps.shape)}",
        f"- source_indices shape: {list(source_indices.shape)}",
        "",
        "Saved NPZ Fields",
        "- X",
        "- timestamps",
        "- source_indices",
        "- feature_names",
        "- seq_len",
        "- stride",
        "- max_time_gap",
    ]

    if allow_time_gap:
        lines.extend(
            [
                "",
                "Notes",
                "- Gap segmentation was disabled by --allow-time-gap, so the full sorted table was used for sliding windows.",
                "- Gap statistics are still reported for inspection.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Notes",
                "- Segments were split at dt <= 0 or dt > max_time_gap.",
                "- Sliding windows were generated only inside each segment.",
            ]
        )

    return lines


def save_npz(
    output_path: Path,
    x: np.ndarray,
    timestamps: np.ndarray,
    source_indices: np.ndarray,
    feature_names: list[str],
    seq_len: int,
    stride: int,
    max_time_gap: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=x,
        timestamps=timestamps,
        source_indices=source_indices,
        feature_names=np.asarray(feature_names, dtype=object),
        seq_len=np.int64(seq_len),
        stride=np.int64(stride),
        max_time_gap=np.float64(max_time_gap),
    )


def write_report(report_path: Path, lines: list[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    validate_args(args)

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()

    ensure_input_exists(input_path)
    feature_names = get_feature_names(args.feature_mode)

    original_df = load_dataframe(input_path)
    validate_columns(original_df)

    total_rows = len(original_df)
    if total_rows < args.seq_len:
        raise SystemExit(
            f"Not enough rows to build sequences: total_rows={total_rows}, "
            f"seq_len={args.seq_len}"
        )

    original_timestamp_info = compute_original_timestamp_info(original_df)
    sorted_df = prepare_sorted_dataframe(original_df, feature_names)
    source_index_column = get_source_index_column(sorted_df)
    sorted_gap_info = compute_sorted_gap_info(sorted_df)
    segments = build_segments(
        num_rows=len(sorted_df),
        sorted_diffs=sorted_gap_info["sorted_diffs"],
        max_time_gap=args.max_time_gap,
        allow_time_gap=args.allow_time_gap,
    )

    total_candidate_windows = count_windows(len(sorted_df), args.seq_len, args.stride)
    x, timestamps, source_indices = build_sequence_arrays(
        sorted_df=sorted_df,
        feature_names=feature_names,
        source_index_column=source_index_column,
        segments=segments,
        seq_len=args.seq_len,
        stride=args.stride,
    )
    kept_windows = int(x.shape[0])

    report_lines = build_report_lines(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        total_rows=total_rows,
        seq_len=args.seq_len,
        stride=args.stride,
        feature_names=feature_names,
        source_index_column=source_index_column,
        max_time_gap=args.max_time_gap,
        allow_time_gap=args.allow_time_gap,
        total_candidate_windows=total_candidate_windows,
        kept_windows=kept_windows,
        segments=segments,
        sorted_gap_info=sorted_gap_info,
        original_timestamp_info=original_timestamp_info,
        x=x,
        timestamps=timestamps,
        source_indices=source_indices,
    )

    save_npz(
        output_path=output_path,
        x=x,
        timestamps=timestamps,
        source_indices=source_indices,
        feature_names=feature_names,
        seq_len=args.seq_len,
        stride=args.stride,
        max_time_gap=args.max_time_gap,
    )
    write_report(report_path, report_lines)

    print("\n".join(report_lines))
    print("")
    print(f"NPZ written to: {output_path}")
    print(f"Report written to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
