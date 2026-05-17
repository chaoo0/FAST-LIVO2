#!/usr/bin/env python3
"""
Analyze and clean exported FAST-LIVO2 + MambaPose training CSV data.

This script is intentionally limited to offline CSV analysis so it does not
change the FAST-LIVO2 runtime path, PoseCompensator inference logic, or ONNX
backend behavior.
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
DEFAULT_INPUT = REPO_ROOT / "Log" / "mamba_pose_train_data.csv"
DEFAULT_OUTPUT_CLEAN = REPO_ROOT / "Log" / "mamba_pose_train_data_clean.csv"
DEFAULT_OUTPUT_REPORT = REPO_ROOT / "Log" / "mamba_pose_train_data_report.txt"

INPUT_FEATURE_COLUMNS = [
    "timestamp",
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
    "history_size",
    "ready_flag",
]

GT_COLUMNS = [
    "gt_pos_x",
    "gt_pos_y",
    "gt_pos_z",
    "gt_rot_x",
    "gt_rot_y",
    "gt_rot_z",
    "gt_rot_w",
]

REQUIRED_COLUMNS = INPUT_FEATURE_COLUMNS + GT_COLUMNS

RANGE_GROUPS = {
    "position_range": ["pos_x", "pos_y", "pos_z"],
    "velocity_range": ["vel_x", "vel_y", "vel_z"],
    "gyro_bias_range": ["bias_g_x", "bias_g_y", "bias_g_z"],
    "accel_bias_range": ["bias_a_x", "bias_a_y", "bias_a_z"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze and clean FAST-LIVO2 + MambaPose exported training CSV data."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-clean",
        type=Path,
        default=DEFAULT_OUTPUT_CLEAN,
        help=f"Clean CSV output path. Default: {DEFAULT_OUTPUT_CLEAN}",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_OUTPUT_REPORT,
        help=f"Text report output path. Default: {DEFAULT_OUTPUT_REPORT}",
    )
    parser.add_argument(
        "--save-clean",
        action="store_true",
        help="Save the cleaned CSV after applying the default valid-sample filter.",
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
        return "1" if value else "0"
    if isinstance(value, (float, np.floating)):
        if math.isinf(float(value)):
            return "inf" if value > 0 else "-inf"
        return f"{float(value):.6f}"
    return str(value)


def summarize_series(series: pd.Series) -> dict[str, float]:
    numeric_series = pd.to_numeric(series, errors="coerce")
    return {
        "min": numeric_series.min(),
        "max": numeric_series.max(),
        "mean": numeric_series.mean(),
    }


def summarize_range(series: pd.Series) -> dict[str, float]:
    numeric_series = pd.to_numeric(series, errors="coerce")
    return {
        "min": numeric_series.min(),
        "max": numeric_series.max(),
    }


def build_nan_inf_mask(df: pd.DataFrame) -> pd.Series:
    numeric_input = df[INPUT_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(numeric_input.to_numpy(dtype=np.float64)).all(axis=1)
    return pd.Series(~finite_mask, index=df.index, name="has_nan_or_inf_in_inputs")


def compute_anomaly_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    nan_inf_mask = build_nan_inf_mask(df)
    return {
        "effective_feature_num_le_0": pd.to_numeric(
            df["effective_feature_num"], errors="coerce"
        )
        <= 0,
        "avg_residual_lt_0": pd.to_numeric(df["avg_residual"], errors="coerce") < 0,
        "ready_flag_eq_0": pd.to_numeric(df["ready_flag"], errors="coerce") == 0,
        "history_size_lt_10": pd.to_numeric(df["history_size"], errors="coerce") < 10,
        "has_nan_or_inf_in_inputs": nan_inf_mask,
    }


def compute_valid_mask(df: pd.DataFrame, anomaly_masks: dict[str, pd.Series]) -> pd.Series:
    return (
        (pd.to_numeric(df["ready_flag"], errors="coerce") == 1)
        & (pd.to_numeric(df["history_size"], errors="coerce") >= 10)
        & (pd.to_numeric(df["effective_feature_num"], errors="coerce") > 0)
        & (pd.to_numeric(df["avg_residual"], errors="coerce") >= 0)
        & (~anomaly_masks["has_nan_or_inf_in_inputs"])
    )


def build_report_lines(
    df: pd.DataFrame,
    input_path: Path,
    output_clean_path: Path,
    output_report_path: Path,
    save_clean: bool,
    anomaly_masks: dict[str, pd.Series],
    valid_mask: pd.Series,
) -> list[str]:
    total_rows = len(df)
    ready_count = int((pd.to_numeric(df["ready_flag"], errors="coerce") == 1).sum())
    ready_ratio = ready_count / total_rows if total_rows else 0.0

    history_stats = summarize_series(df["history_size"])
    feature_stats = summarize_series(df["effective_feature_num"])
    residual_stats = summarize_series(df["avg_residual"])

    valid_count = int(valid_mask.sum())
    valid_ratio = valid_count / total_rows if total_rows else 0.0

    lines = [
        "Mamba Pose Training Data Analysis Report",
        "=" * 40,
        "",
        "Input / Output",
        f"- input_csv: {input_path}",
        f"- output_report: {output_report_path}",
        f"- output_clean_csv: {output_clean_path}",
        f"- save_clean_enabled: {'true' if save_clean else 'false'}",
        "",
        "Dataset Overview",
        f"- total_rows: {total_rows}",
        f"- columns ({len(df.columns)}): {', '.join(df.columns)}",
        f"- ready_flag_eq_1: {ready_count} ({ready_ratio:.2%})",
        "",
        "Core Statistics",
        (
            "- history_size: "
            f"min={format_scalar(history_stats['min'])}, "
            f"max={format_scalar(history_stats['max'])}, "
            f"mean={format_scalar(history_stats['mean'])}"
        ),
        (
            "- effective_feature_num: "
            f"min={format_scalar(feature_stats['min'])}, "
            f"max={format_scalar(feature_stats['max'])}, "
            f"mean={format_scalar(feature_stats['mean'])}"
        ),
        (
            "- avg_residual: "
            f"min={format_scalar(residual_stats['min'])}, "
            f"max={format_scalar(residual_stats['max'])}, "
            f"mean={format_scalar(residual_stats['mean'])}"
        ),
    ]

    for group_name, columns in RANGE_GROUPS.items():
        lines.append(f"- {group_name}:")
        for column in columns:
            range_stats = summarize_range(df[column])
            lines.append(
                f"  - {column}: min={format_scalar(range_stats['min'])}, "
                f"max={format_scalar(range_stats['max'])}"
            )

    lines.extend(
        [
            "",
            "Anomaly Frame Counts",
        ]
    )

    for name, mask in anomaly_masks.items():
        count = int(mask.sum())
        ratio = count / total_rows if total_rows else 0.0
        lines.append(f"- {name}: {count} ({ratio:.2%})")

    lines.extend(
        [
            "",
            "Default Valid Sample Filter",
            "- ready_flag == 1",
            "- history_size >= 10",
            "- effective_feature_num > 0",
            "- avg_residual >= 0",
            "- no NaN / Inf in input feature fields",
            (
                f"- valid_rows_after_filter: {valid_count} "
                f"({valid_ratio:.2%})"
            ),
            "",
            "Training Usage Notes",
            (
                "- The cleaned CSV keeps the original column structure and only "
                "removes invalid rows."
            ),
            (
                "- Use the cleaned CSV as the base table for later fixed-length "
                "sequence slicing, typically T=10 or larger."
            ),
            (
                "- Current gt_* fields may remain NaN and should not be treated "
                "as ready-made supervised labels."
            ),
            (
                "- For real Mamba Pose Correction training, build temporal "
                "windows from consecutive clean rows and add ground-truth or "
                "pseudo-label alignment in the training pipeline."
            ),
        ]
    )

    if save_clean:
        lines.extend(
            [
                "",
                "Saved Outputs",
                f"- cleaned_csv_written: {output_clean_path}",
                f"- cleaned_row_count: {valid_count}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Saved Outputs",
                "- cleaned_csv_written: skipped (--save-clean not set)",
            ]
        )

    return lines


def write_report(report_path: Path, lines: list[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_clean_csv(clean_path: Path, clean_df: pd.DataFrame) -> None:
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(clean_path, index=False)


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_clean_path = args.output_clean.expanduser().resolve()
    output_report_path = args.output_report.expanduser().resolve()

    ensure_input_exists(input_path)
    df = load_dataframe(input_path)
    validate_columns(df)

    anomaly_masks = compute_anomaly_masks(df)
    valid_mask = compute_valid_mask(df, anomaly_masks)
    clean_df = df.loc[valid_mask].copy()

    report_lines = build_report_lines(
        df=df,
        input_path=input_path,
        output_clean_path=output_clean_path,
        output_report_path=output_report_path,
        save_clean=args.save_clean,
        anomaly_masks=anomaly_masks,
        valid_mask=valid_mask,
    )

    print("\n".join(report_lines))
    write_report(output_report_path, report_lines)

    if args.save_clean:
        save_clean_csv(output_clean_path, clean_df)

    print("")
    print(f"Report written to: {output_report_path}")
    if args.save_clean:
        print(f"Clean CSV written to: {output_clean_path}")
    else:
        print("Clean CSV not written. Use --save-clean to save it.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
