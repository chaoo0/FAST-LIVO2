#!/usr/bin/env python3
"""
Build a static zero-label baseline dataset from normalized Mamba pose sequences.

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


np = require_dependency("numpy", "python3 -m pip install numpy")


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "Log" / "mamba_pose_sequences_T10_interpolated_normalized.npz"
DEFAULT_NORM = REPO_ROOT / "Log" / "mamba_pose_feature_norm_T10_interpolated.npz"
DEFAULT_OUTPUT = REPO_ROOT / "Log" / "mamba_pose_static_zero_label_dataset_T10.npz"
DEFAULT_REPORT = (
    REPO_ROOT / "Log" / "mamba_pose_static_zero_label_dataset_T10_report.txt"
)

EXPECTED_FEATURE_NAMES = [
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

LABEL_NAMES = [
    "d_roll",
    "d_pitch",
    "d_yaw",
    "d_tx",
    "d_ty",
    "d_tz",
]

REQUIRED_INPUT_FIELDS = [
    "X_normalized",
    "timestamps",
    "source_indices",
    "feature_names",
    "seq_len",
    "stride",
    "max_time_gap",
    "norm_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a static zero-label baseline dataset from normalized Mamba "
            "pose sequence NPZ data."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input normalized sequence NPZ path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--norm",
        type=Path,
        default=DEFAULT_NORM,
        help=f"Input feature norm NPZ path. Default: {DEFAULT_NORM}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output dataset NPZ path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Output TXT report path. Default: {DEFAULT_REPORT}",
    )
    parser.add_argument(
        "--label-type",
        default="static_zero",
        help="Label type to build. Only 'static_zero' is supported right now.",
    )
    return parser.parse_args()


def ensure_input_exists(input_path: Path, description: str) -> None:
    if not input_path.exists():
        raise SystemExit(f"{description} not found: {input_path}")
    if not input_path.is_file():
        raise SystemExit(f"{description} is not a file: {input_path}")


def format_scalar(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return value
    if value is None:
        return "none"
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.6f}"
    return str(value)


def load_npz(input_path: Path):
    try:
        return np.load(input_path, allow_pickle=True)
    except Exception as exc:
        raise SystemExit(f"Failed to load NPZ '{input_path}': {exc}") from exc


def extract_scalar(array_like, field_name: str) -> int | float:
    array_value = np.asarray(array_like)
    if array_value.size != 1:
        raise SystemExit(
            f"Field '{field_name}' must be a scalar in the input NPZ. "
            f"Got shape {array_value.shape}."
        )
    return array_value.reshape(()).item()


def decode_string_array(array_like) -> list[str]:
    values = np.asarray(array_like).tolist()
    decoded: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def validate_args(args: argparse.Namespace) -> None:
    if args.label_type != "static_zero":
        raise SystemExit(
            f"Unsupported --label-type: {args.label_type}. "
            "Only 'static_zero' is supported in this baseline script."
        )


def validate_input_fields(npz_data) -> None:
    missing_fields = [field for field in REQUIRED_INPUT_FIELDS if field not in npz_data]
    if missing_fields:
        raise SystemExit(
            "Input normalized NPZ is missing required fields:\n"
            + "\n".join(f"  - {field}" for field in missing_fields)
        )


def validate_normalized_input(npz_data) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int, int, float, str]:
    x_normalized = np.asarray(npz_data["X_normalized"])
    timestamps = np.asarray(npz_data["timestamps"])
    source_indices = np.asarray(npz_data["source_indices"])
    feature_names = decode_string_array(npz_data["feature_names"])
    seq_len = int(extract_scalar(npz_data["seq_len"], "seq_len"))
    stride = int(extract_scalar(npz_data["stride"], "stride"))
    max_time_gap = float(extract_scalar(npz_data["max_time_gap"], "max_time_gap"))
    norm_path_recorded = str(extract_scalar(npz_data["norm_path"], "norm_path"))

    if x_normalized.ndim != 3:
        raise SystemExit(
            "Input X_normalized must have shape [N, T, 18]. "
            f"Got shape {x_normalized.shape}."
        )
    if x_normalized.shape[2] != 18:
        raise SystemExit(
            f"Input X_normalized must have feature_dim=18. Got shape {x_normalized.shape}."
        )
    if x_normalized.shape[0] <= 0:
        raise SystemExit(
            "Input X_normalized has zero sequences. At least one sequence is required."
        )
    if x_normalized.shape[1] != seq_len:
        raise SystemExit(
            f"Input X_normalized second dimension {x_normalized.shape[1]} "
            f"does not match seq_len={seq_len}."
        )
    if timestamps.shape != x_normalized.shape[:2]:
        raise SystemExit(
            "Input timestamps must match X_normalized [N, T]. "
            f"Got {timestamps.shape} vs expected {x_normalized.shape[:2]}."
        )
    if source_indices.shape != x_normalized.shape[:2]:
        raise SystemExit(
            "Input source_indices must match X_normalized [N, T]. "
            f"Got {source_indices.shape} vs expected {x_normalized.shape[:2]}."
        )
    if len(feature_names) != 18:
        raise SystemExit(
            f"Input feature_names must contain 18 names. Got {len(feature_names)}."
        )
    if feature_names != EXPECTED_FEATURE_NAMES:
        raise SystemExit(
            "Input feature_names do not match the required basic18 order.\n"
            f"Expected: {', '.join(EXPECTED_FEATURE_NAMES)}\n"
            f"Got: {', '.join(feature_names)}"
        )

    x_float = x_normalized.astype(np.float32, copy=False)
    x_nan_count = int(np.isnan(x_float).sum())
    x_inf_count = int(np.isinf(x_float).sum())
    if x_nan_count > 0 or x_inf_count > 0:
        raise SystemExit(
            "Input X_normalized contains non-finite values. "
            f"NaN count: {x_nan_count}, Inf count: {x_inf_count}"
        )

    return (
        x_float,
        timestamps,
        source_indices,
        feature_names,
        seq_len,
        stride,
        max_time_gap,
        norm_path_recorded,
    )


def validate_norm_npz(norm_npz_data, norm_path: Path, feature_names: list[str]) -> None:
    required_norm_fields = ["feature_names", "mean", "std_safe"]
    missing_fields = [field for field in required_norm_fields if field not in norm_npz_data]
    if missing_fields:
        raise SystemExit(
            f"Norm NPZ '{norm_path}' is missing required fields:\n"
            + "\n".join(f"  - {field}" for field in missing_fields)
        )

    norm_feature_names = decode_string_array(norm_npz_data["feature_names"])
    if norm_feature_names != feature_names:
        raise SystemExit(
            "Norm NPZ feature_names do not match input normalized dataset feature_names.\n"
            f"Norm: {', '.join(norm_feature_names)}\n"
            f"Input: {', '.join(feature_names)}"
        )


def build_zero_labels(num_sequences: int) -> np.ndarray:
    y = np.zeros((num_sequences, len(LABEL_NAMES)), dtype=np.float32)
    y_nan_count = int(np.isnan(y).sum())
    y_inf_count = int(np.isinf(y).sum())
    if y_nan_count > 0 or y_inf_count > 0:
        raise SystemExit(
            "Generated y contains non-finite values, which should never happen. "
            f"NaN count: {y_nan_count}, Inf count: {y_inf_count}"
        )
    if not np.all(y == 0.0):
        raise SystemExit("Generated static_zero labels are not all zero as expected.")
    return y


def build_report_lines(
    args: argparse.Namespace,
    x_normalized: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    norm_path_recorded: str,
    x_nan_count: int,
    x_inf_count: int,
    y_nan_count: int,
    y_inf_count: int,
) -> list[str]:
    return [
        "Mamba Pose Static Zero-Label Dataset Report",
        "=" * 45,
        "",
        "Input / Output",
        f"- input_npz: {args.input}",
        f"- norm_npz: {args.norm}",
        f"- output_npz: {args.output}",
        f"- output_report: {args.report}",
        f"- recorded_norm_path_from_input_npz: {norm_path_recorded}",
        "",
        "Dataset Summary",
        f"- X_normalized shape: {list(x_normalized.shape)}",
        f"- y shape: {list(y.shape)}",
        f"- num_sequences: {x_normalized.shape[0]}",
        f"- seq_len: {x_normalized.shape[1]}",
        f"- feature_dim: {x_normalized.shape[2]}",
        f"- label_dim: {y.shape[1]}",
        f"- feature_names: {', '.join(feature_names)}",
        f"- label_names: {', '.join(LABEL_NAMES)}",
        f"- label_type: {args.label_type}",
        "",
        "Finite Checks",
        f"- X_normalized_nan_count: {x_nan_count}",
        f"- X_normalized_inf_count: {x_inf_count}",
        f"- y_nan_count: {y_nan_count}",
        f"- y_inf_count: {y_inf_count}",
        "",
        "y Statistics",
        f"- y_min: {format_scalar(float(np.min(y)))}",
        f"- y_max: {format_scalar(float(np.max(y)))}",
        f"- y_mean: {format_scalar(float(np.mean(y)))}",
        f"- y_std: {format_scalar(float(np.std(y)))}",
        "",
        "Warnings",
        "- Current y = 0 is only valid as a static zero-label no-op baseline for a stationary rosbag.",
        "- This dataset is only meant to validate training, dataloader, loss, ONNX export, and FAST-LIVO2 inference loop integration later.",
        "- This is not the final real pose correction label and cannot prove dynamic pose compensation ability.",
        "",
        "How To Load In Training",
        "- Load X_normalized as the model input and y as the 6D correction target.",
        "- Keep label_names and feature_names together with the dataset so the training code can assert the same ordering.",
        "- Keep using the same norm NPZ recorded in norm_path for any later preprocessing checks.",
    ]


def main() -> None:
    args = parse_args()
    validate_args(args)
    ensure_input_exists(args.input, "Input normalized NPZ")
    ensure_input_exists(args.norm, "Input norm NPZ")

    input_npz = load_npz(args.input)
    validate_input_fields(input_npz)
    (
        x_normalized,
        timestamps,
        source_indices,
        feature_names,
        seq_len,
        stride,
        max_time_gap,
        norm_path_recorded,
    ) = validate_normalized_input(input_npz)

    norm_npz = load_npz(args.norm)
    validate_norm_npz(norm_npz, args.norm, feature_names)

    y = build_zero_labels(x_normalized.shape[0])

    x_nan_count = int(np.isnan(x_normalized).sum())
    x_inf_count = int(np.isinf(x_normalized).sum())
    y_nan_count = int(np.isnan(y).sum())
    y_inf_count = int(np.isinf(y).sum())

    report_lines = build_report_lines(
        args=args,
        x_normalized=x_normalized,
        y=y,
        feature_names=feature_names,
        norm_path_recorded=norm_path_recorded,
        x_nan_count=x_nan_count,
        x_inf_count=x_inf_count,
        y_nan_count=y_nan_count,
        y_inf_count=y_inf_count,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        args.output,
        X_normalized=x_normalized.astype(np.float32, copy=False),
        y=y.astype(np.float32, copy=False),
        timestamps=timestamps,
        source_indices=source_indices,
        feature_names=np.asarray(feature_names, dtype=object),
        label_names=np.asarray(LABEL_NAMES, dtype=object),
        seq_len=np.asarray(seq_len, dtype=np.int64),
        stride=np.asarray(stride, dtype=np.int64),
        max_time_gap=np.asarray(max_time_gap, dtype=np.float64),
        norm_path=np.asarray(str(args.norm), dtype=object),
        label_type=np.asarray(args.label_type, dtype=object),
        input_path=np.asarray(str(args.input), dtype=object),
        num_sequences=np.asarray(x_normalized.shape[0], dtype=np.int64),
        feature_dim=np.asarray(x_normalized.shape[2], dtype=np.int64),
        label_dim=np.asarray(len(LABEL_NAMES), dtype=np.int64),
    )

    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
