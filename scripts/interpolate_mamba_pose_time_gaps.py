#!/usr/bin/env python3
"""
Interpolate short timestamp gaps in cleaned FAST-LIVO2 + MambaPose CSV data.

This script is intentionally limited to offline data preparation. It does not
modify FAST-LIVO2 runtime code, PoseCompensator inference logic, ONNX backend
behavior, the safety layer, launch files, or YAML configuration.
"""

from __future__ import annotations

import argparse
import math
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
DEFAULT_OUTPUT = REPO_ROOT / "Log" / "mamba_pose_train_data_interpolated.csv"
DEFAULT_REPORT = REPO_ROOT / "Log" / "mamba_pose_train_data_interpolated_report.txt"

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

LINEAR_FEATURE_NAMES = [
    "pos_x",
    "pos_y",
    "pos_z",
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

QUATERNION_COLUMNS = ["rot_x", "rot_y", "rot_z", "rot_w"]
GT_COLUMNS = [
    "gt_pos_x",
    "gt_pos_y",
    "gt_pos_z",
    "gt_rot_x",
    "gt_rot_y",
    "gt_rot_z",
    "gt_rot_w",
]
REQUIRED_COLUMNS = (
    ["timestamp"]
    + BASIC18_FEATURE_NAMES
    + ["history_size", "ready_flag"]
    + GT_COLUMNS
)
TRACKING_COLUMNS = [
    "is_interpolated",
    "interp_alpha",
    "interp_left_timestamp",
    "interp_right_timestamp",
    "interp_left_index",
    "interp_right_index",
    "interp_method",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate only short timestamp gaps in cleaned FAST-LIVO2 + "
            "MambaPose CSV data for later fixed-length sequence building."
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
        help=f"Interpolated CSV output path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Interpolation TXT report path. Default: {DEFAULT_REPORT}",
    )
    parser.add_argument(
        "--target-dt",
        default="auto",
        help=(
            "Target interpolation delta time in seconds, or 'auto'. "
            "Default: auto"
        ),
    )
    parser.add_argument(
        "--max-continuous-gap",
        type=float,
        default=0.2,
        help="Gap <= this value is treated as continuous. Default: 0.2",
    )
    parser.add_argument(
        "--max-interpolate-gap",
        type=float,
        default=0.5,
        help=(
            "Gap > max_continuous_gap and <= this value is eligible for "
            "interpolation. Default: 0.5"
        ),
    )
    parser.add_argument(
        "--max-insert-per-gap",
        type=int,
        default=5,
        help="Maximum inserted rows per eligible gap. Default: 5",
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
    if args.max_continuous_gap <= 0:
        raise SystemExit("--max-continuous-gap must be positive.")
    if args.max_interpolate_gap <= 0:
        raise SystemExit("--max-interpolate-gap must be positive.")
    if args.max_interpolate_gap < args.max_continuous_gap:
        raise SystemExit(
            "--max-interpolate-gap must be >= --max-continuous-gap."
        )
    if args.max_insert_per_gap <= 0:
        raise SystemExit("--max-insert-per-gap must be positive.")


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


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 2:
        raise SystemExit(
            "Input CSV has fewer than 2 rows. At least 2 rows are required "
            "to analyze timestamp gaps."
        )

    prepared = df.copy()
    prepared["input_row_index"] = np.arange(len(prepared), dtype=np.int64)

    for column in REQUIRED_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    finite_required_columns = ["timestamp"] + BASIC18_FEATURE_NAMES + [
        "history_size",
        "ready_flag",
    ]
    invalid_numeric_rows = ~np.isfinite(
        prepared[finite_required_columns].to_numpy(dtype=np.float64)
    ).all(axis=1)
    if invalid_numeric_rows.any():
        count = int(invalid_numeric_rows.sum())
        raise SystemExit(
            "Input clean CSV contains non-finite values in required numeric "
            f"fields. Invalid row count: {count}"
        )

    quaternion_array = prepared[QUATERNION_COLUMNS].to_numpy(dtype=np.float64)
    quaternion_norms = np.linalg.norm(quaternion_array, axis=1)
    bad_norm_mask = ~np.isfinite(quaternion_norms) | (quaternion_norms <= 1e-12)
    if bad_norm_mask.any():
        count = int(bad_norm_mask.sum())
        raise SystemExit(
            "Input clean CSV contains invalid quaternions with zero or "
            f"non-finite norm. Invalid row count: {count}"
        )

    return prepared.sort_values(
        by=["timestamp", "input_row_index"], kind="mergesort"
    ).reset_index(drop=True)


def resolve_target_dt(
    target_dt_arg: str,
    sorted_diffs: np.ndarray,
    max_continuous_gap: float,
    max_interpolate_gap: float,
) -> tuple[float, str]:
    if target_dt_arg != "auto":
        try:
            target_dt = float(target_dt_arg)
        except ValueError as exc:
            raise SystemExit(
                f"Invalid --target-dt value: {target_dt_arg}. Use 'auto' or a positive float."
            ) from exc
        if not math.isfinite(target_dt) or target_dt <= 0:
            raise SystemExit("--target-dt must be a positive finite float or 'auto'.")
        return target_dt, "manual"

    positive_diffs = sorted_diffs[sorted_diffs > 0]
    if positive_diffs.size == 0:
        raise SystemExit(
            "Unable to resolve target dt automatically because the CSV has no "
            "positive timestamp gaps."
        )

    continuous_diffs = positive_diffs[positive_diffs <= max_continuous_gap]
    if continuous_diffs.size > 0:
        return float(np.median(continuous_diffs)), "auto_median_positive_continuous_gap"

    short_diffs = positive_diffs[positive_diffs <= max_interpolate_gap]
    if short_diffs.size > 0:
        return float(np.median(short_diffs)), "auto_median_positive_short_gap"

    return float(np.median(positive_diffs)), "auto_median_positive_gap"


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise SystemExit(
            "Encountered an invalid quaternion during interpolation. "
            "Quaternion norm must be finite and > 0."
        )
    return quaternion / norm


def quaternion_nlerp(
    left_quaternion: np.ndarray, right_quaternion: np.ndarray, alpha: float
) -> np.ndarray:
    q0 = normalize_quaternion(left_quaternion.astype(np.float64, copy=False))
    q1 = normalize_quaternion(right_quaternion.astype(np.float64, copy=False))
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    blended = (1.0 - alpha) * q0 + alpha * q1
    return normalize_quaternion(blended)


def build_insert_timestamps(
    left_timestamp: float,
    right_timestamp: float,
    target_dt: float,
    max_insert_per_gap: int,
) -> tuple[np.ndarray, bool, str]:
    gap = right_timestamp - left_timestamp
    tolerance = max(1e-12, 1e-9 * max(abs(gap), abs(target_dt), 1.0))
    candidate_times: list[float] = []
    next_timestamp = left_timestamp + target_dt
    while next_timestamp < right_timestamp - tolerance:
        candidate_times.append(float(next_timestamp))
        next_timestamp += target_dt
        if len(candidate_times) > 100000:
            raise SystemExit(
                "Interpolation candidate count grew unexpectedly large. "
                "Check the timestamp scale and --target-dt value."
            )

    if not candidate_times:
        return np.array([], dtype=np.float64), False, "none"

    if len(candidate_times) <= max_insert_per_gap:
        return np.array(candidate_times, dtype=np.float64), False, "target_dt"

    limited_times = np.linspace(
        left_timestamp,
        right_timestamp,
        num=max_insert_per_gap + 2,
        dtype=np.float64,
    )[1:-1]
    return limited_times, True, "uniform_limited"


def create_original_row(row: pd.Series, output_columns: list[str]) -> dict[str, object]:
    output_row = {column: row[column] if column in row.index else np.nan for column in output_columns}
    output_row["is_interpolated"] = 0
    output_row["interp_alpha"] = 0.0
    output_row["interp_left_timestamp"] = np.nan
    output_row["interp_right_timestamp"] = np.nan
    output_row["interp_left_index"] = np.nan
    output_row["interp_right_index"] = np.nan
    output_row["interp_method"] = "original"
    return output_row


def create_interpolated_row(
    left_row: pd.Series,
    right_row: pd.Series,
    timestamp: float,
    alpha: float,
    interp_method: str,
    output_columns: list[str],
) -> dict[str, object]:
    output_row = {column: np.nan for column in output_columns}

    output_row["timestamp"] = float(timestamp)
    for column in LINEAR_FEATURE_NAMES:
        left_value = float(left_row[column])
        right_value = float(right_row[column])
        output_row[column] = (1.0 - alpha) * left_value + alpha * right_value

    quaternion = quaternion_nlerp(
        left_row[QUATERNION_COLUMNS].to_numpy(dtype=np.float64),
        right_row[QUATERNION_COLUMNS].to_numpy(dtype=np.float64),
        alpha,
    )
    for column, value in zip(QUATERNION_COLUMNS, quaternion):
        output_row[column] = float(value)

    output_row["history_size"] = max(
        float(left_row["history_size"]),
        float(right_row["history_size"]),
        10.0,
    )
    output_row["ready_flag"] = 1.0

    for column in GT_COLUMNS:
        output_row[column] = np.nan

    output_row["is_interpolated"] = 1
    output_row["interp_alpha"] = float(alpha)
    output_row["interp_left_timestamp"] = float(left_row["timestamp"])
    output_row["interp_right_timestamp"] = float(right_row["timestamp"])
    output_row["interp_left_index"] = int(left_row["input_row_index"])
    output_row["interp_right_index"] = int(right_row["input_row_index"])
    output_row["interp_method"] = interp_method
    return output_row


def interpolate_dataframe(
    sorted_df: pd.DataFrame,
    target_dt: float,
    max_continuous_gap: float,
    max_interpolate_gap: float,
    max_insert_per_gap: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    output_columns = [
        column for column in sorted_df.columns if column != "input_row_index"
    ] + TRACKING_COLUMNS

    timestamps = sorted_df["timestamp"].to_numpy(dtype=np.float64)
    sorted_diffs = np.diff(timestamps)

    gap_stats = {
        "non_positive_gap_count": 0,
        "continuous_gap_count": 0,
        "interpolatable_gap_count": 0,
        "actually_interpolated_gap_count": 0,
        "long_gap_count": 0,
        "limited_by_max_insert_per_gap_count": 0,
        "inserted_rows": 0,
    }

    output_rows: list[dict[str, object]] = []

    for index in range(len(sorted_df) - 1):
        left_row = sorted_df.iloc[index]
        right_row = sorted_df.iloc[index + 1]
        output_rows.append(create_original_row(left_row, output_columns))

        dt = float(sorted_diffs[index])
        if dt <= 0:
            gap_stats["non_positive_gap_count"] += 1
            continue
        if dt <= max_continuous_gap:
            gap_stats["continuous_gap_count"] += 1
            continue
        if dt > max_interpolate_gap:
            gap_stats["long_gap_count"] += 1
            continue

        gap_stats["interpolatable_gap_count"] += 1
        insert_timestamps, was_limited, insert_mode = build_insert_timestamps(
            left_timestamp=float(left_row["timestamp"]),
            right_timestamp=float(right_row["timestamp"]),
            target_dt=target_dt,
            max_insert_per_gap=max_insert_per_gap,
        )
        if was_limited:
            gap_stats["limited_by_max_insert_per_gap_count"] += 1
        if insert_timestamps.size == 0:
            continue

        gap_stats["actually_interpolated_gap_count"] += 1
        gap_stats["inserted_rows"] += int(insert_timestamps.size)

        for timestamp in insert_timestamps:
            alpha = (float(timestamp) - float(left_row["timestamp"])) / dt
            alpha = min(max(alpha, 0.0), 1.0)
            interp_method = (
                "linear+nlerp"
                if insert_mode == "target_dt"
                else "linear+nlerp_uniform_limited"
            )
            output_rows.append(
                create_interpolated_row(
                    left_row=left_row,
                    right_row=right_row,
                    timestamp=float(timestamp),
                    alpha=float(alpha),
                    interp_method=interp_method,
                    output_columns=output_columns,
                )
            )

    output_rows.append(create_original_row(sorted_df.iloc[-1], output_columns))

    output_df = pd.DataFrame(output_rows, columns=output_columns)
    output_df = output_df.sort_values(
        by=["timestamp", "is_interpolated", "interp_alpha"],
        kind="mergesort",
    ).reset_index(drop=True)

    return output_df, gap_stats


def compute_gap_stats(timestamps: np.ndarray) -> tuple[float, float]:
    if timestamps.size <= 1:
        return 0.0, 0.0
    diffs = np.diff(timestamps)
    return float(diffs.max()), float(diffs.mean())


def compute_basic18_non_finite_counts(output_df: pd.DataFrame) -> tuple[int, int]:
    basic18_array = output_df[BASIC18_FEATURE_NAMES].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(basic18_array)
    non_finite_cell_count = int((~finite_mask).sum())
    non_finite_row_count = int((~finite_mask.all(axis=1)).sum())
    return non_finite_cell_count, non_finite_row_count


def compute_quaternion_norm_error_stats(
    output_df: pd.DataFrame,
) -> tuple[float, float, float]:
    quaternion_array = output_df[QUATERNION_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    norms = np.linalg.norm(quaternion_array, axis=1)
    errors = np.abs(norms - 1.0)
    return float(errors.min()), float(errors.max()), float(errors.mean())


def build_report_lines(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    original_rows: int,
    output_rows: int,
    inserted_rows: int,
    target_dt: float,
    target_dt_mode: str,
    max_continuous_gap: float,
    max_interpolate_gap: float,
    max_insert_per_gap: int,
    gap_stats: dict[str, object],
    max_gap_before: float,
    mean_gap_before: float,
    max_gap_after: float,
    mean_gap_after: float,
    non_finite_basic18_cells: int,
    non_finite_basic18_rows: int,
    quaternion_error_min: float,
    quaternion_error_max: float,
    quaternion_error_mean: float,
) -> list[str]:
    interpolated_row_ratio = (inserted_rows / output_rows) if output_rows else 0.0

    return [
        "Mamba Pose Time Gap Interpolation Report",
        "=" * 39,
        "",
        "Input / Output",
        f"- input_csv: {input_path}",
        f"- output_csv: {output_path}",
        f"- output_report: {report_path}",
        "",
        "Core Counts",
        f"- original_rows: {original_rows}",
        f"- output_rows: {output_rows}",
        f"- inserted_rows: {inserted_rows}",
        f"- interpolated_row_ratio: {interpolated_row_ratio:.2%}",
        "",
        "Interpolation Parameters",
        f"- target_dt: {format_scalar(target_dt)}",
        f"- target_dt_mode: {target_dt_mode}",
        f"- max_continuous_gap: {format_scalar(max_continuous_gap)}",
        f"- max_interpolate_gap: {format_scalar(max_interpolate_gap)}",
        f"- max_insert_per_gap: {max_insert_per_gap}",
        "",
        "Gap Classification",
        f"- non_positive_gap_count: {gap_stats['non_positive_gap_count']}",
        f"- continuous_gap_count: {gap_stats['continuous_gap_count']}",
        f"- interpolatable_gap_count: {gap_stats['interpolatable_gap_count']}",
        f"- actually_interpolated_gap_count: {gap_stats['actually_interpolated_gap_count']}",
        f"- long_gap_count: {gap_stats['long_gap_count']}",
        (
            "- limited_by_max_insert_per_gap_count: "
            f"{gap_stats['limited_by_max_insert_per_gap_count']}"
        ),
        "",
        "Gap Statistics",
        f"- max_gap_before: {format_scalar(max_gap_before)}",
        f"- mean_gap_before: {format_scalar(mean_gap_before)}",
        f"- max_gap_after: {format_scalar(max_gap_after)}",
        f"- mean_gap_after: {format_scalar(mean_gap_after)}",
        "",
        "Output Quality Checks",
        f"- basic18_nan_or_inf_count: {non_finite_basic18_cells}",
        f"- basic18_nan_or_inf_row_count: {non_finite_basic18_rows}",
        (
            "- quaternion_norm_error_abs: "
            f"min={format_scalar(quaternion_error_min)}, "
            f"max={format_scalar(quaternion_error_max)}, "
            f"mean={format_scalar(quaternion_error_mean)}"
        ),
        "",
        "Recommended Next Command",
        (
            "- python3 scripts/build_mamba_pose_sequences.py "
            "--input Log/mamba_pose_train_data_interpolated.csv "
            "--output Log/mamba_pose_sequences_T10_from_interpolated.npz "
            "--report Log/mamba_pose_sequences_T10_from_interpolated_report.txt"
        ),
    ]


def main() -> None:
    args = parse_args()
    validate_args(args)
    ensure_input_exists(args.input)

    input_df = load_dataframe(args.input)
    validate_columns(input_df)
    sorted_df = prepare_dataframe(input_df)

    original_timestamps = sorted_df["timestamp"].to_numpy(dtype=np.float64)
    sorted_diffs = np.diff(original_timestamps)
    target_dt, target_dt_mode = resolve_target_dt(
        target_dt_arg=args.target_dt,
        sorted_diffs=sorted_diffs,
        max_continuous_gap=args.max_continuous_gap,
        max_interpolate_gap=args.max_interpolate_gap,
    )

    output_df, gap_stats = interpolate_dataframe(
        sorted_df=sorted_df,
        target_dt=target_dt,
        max_continuous_gap=args.max_continuous_gap,
        max_interpolate_gap=args.max_interpolate_gap,
        max_insert_per_gap=args.max_insert_per_gap,
    )

    output_timestamps = output_df["timestamp"].to_numpy(dtype=np.float64)
    max_gap_before, mean_gap_before = compute_gap_stats(original_timestamps)
    max_gap_after, mean_gap_after = compute_gap_stats(output_timestamps)
    non_finite_basic18_cells, non_finite_basic18_rows = compute_basic18_non_finite_counts(
        output_df
    )
    quaternion_error_min, quaternion_error_max, quaternion_error_mean = (
        compute_quaternion_norm_error_stats(output_df)
    )

    report_lines = build_report_lines(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        original_rows=len(sorted_df),
        output_rows=len(output_df),
        inserted_rows=int(gap_stats["inserted_rows"]),
        target_dt=target_dt,
        target_dt_mode=target_dt_mode,
        max_continuous_gap=args.max_continuous_gap,
        max_interpolate_gap=args.max_interpolate_gap,
        max_insert_per_gap=args.max_insert_per_gap,
        gap_stats=gap_stats,
        max_gap_before=max_gap_before,
        mean_gap_before=mean_gap_before,
        max_gap_after=max_gap_after,
        mean_gap_after=mean_gap_after,
        non_finite_basic18_cells=non_finite_basic18_cells,
        non_finite_basic18_rows=non_finite_basic18_rows,
        quaternion_error_min=quaternion_error_min,
        quaternion_error_max=quaternion_error_max,
        quaternion_error_mean=quaternion_error_mean,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
