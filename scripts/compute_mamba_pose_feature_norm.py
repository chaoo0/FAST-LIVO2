#!/usr/bin/env python3
"""
Compute feature normalization statistics from T=10 Mamba pose sequence NPZ data.

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
DEFAULT_INPUT = REPO_ROOT / "Log" / "mamba_pose_sequences_T10_interpolated.npz"
DEFAULT_OUTPUT_NORM = REPO_ROOT / "Log" / "mamba_pose_feature_norm_T10_interpolated.npz"
DEFAULT_OUTPUT_REPORT = (
    REPO_ROOT / "Log" / "mamba_pose_feature_norm_T10_interpolated_report.txt"
)
DEFAULT_OUTPUT_NORMALIZED = (
    REPO_ROOT / "Log" / "mamba_pose_sequences_T10_interpolated_normalized.npz"
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

REQUIRED_NPZ_FIELDS = [
    "X",
    "timestamps",
    "source_indices",
    "feature_names",
    "seq_len",
    "stride",
    "max_time_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute feature normalization statistics for later Mamba pose "
            "model training from a sequence NPZ."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input sequence NPZ path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-norm",
        type=Path,
        default=DEFAULT_OUTPUT_NORM,
        help=f"Output norm NPZ path. Default: {DEFAULT_OUTPUT_NORM}",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_OUTPUT_REPORT,
        help=f"Output TXT report path. Default: {DEFAULT_OUTPUT_REPORT}",
    )
    parser.add_argument(
        "--output-normalized",
        type=Path,
        default=DEFAULT_OUTPUT_NORMALIZED,
        help=(
            "Optional normalized sequence NPZ path used when --save-normalized "
            f"is set. Default: {DEFAULT_OUTPUT_NORMALIZED}"
        ),
    )
    parser.add_argument(
        "--save-normalized",
        action="store_true",
        help="Save an additional normalized sequence NPZ.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Minimum std threshold before std_safe falls back to 1.0. Default: 1e-6",
    )
    parser.add_argument(
        "--clip-percentile-low",
        type=float,
        default=1.0,
        help="Reference low percentile for clipping diagnostics. Default: 1",
    )
    parser.add_argument(
        "--clip-percentile-high",
        type=float,
        default=99.0,
        help="Reference high percentile for clipping diagnostics. Default: 99",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epsilon <= 0 or not math.isfinite(args.epsilon):
        raise SystemExit("--epsilon must be a positive finite float.")
    if not (0.0 <= args.clip_percentile_low < args.clip_percentile_high <= 100.0):
        raise SystemExit(
            "--clip-percentile-low and --clip-percentile-high must satisfy "
            "0 <= low < high <= 100."
        )


def ensure_input_exists(input_path: Path) -> None:
    if not input_path.exists():
        raise SystemExit(f"Input NPZ not found: {input_path}")
    if not input_path.is_file():
        raise SystemExit(f"Input path is not a file: {input_path}")


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


def validate_npz_fields(npz_data) -> None:
    missing_fields = [field for field in REQUIRED_NPZ_FIELDS if field not in npz_data]
    if missing_fields:
        raise SystemExit(
            "Input NPZ is missing required fields:\n"
            + "\n".join(f"  - {field}" for field in missing_fields)
        )


def decode_feature_names(feature_names_array: np.ndarray) -> list[str]:
    names: list[str] = []
    for item in feature_names_array.tolist():
        if isinstance(item, bytes):
            names.append(item.decode("utf-8"))
        else:
            names.append(str(item))
    return names


def extract_scalar(array_like, field_name: str) -> int | float:
    array_value = np.asarray(array_like)
    if array_value.size != 1:
        raise SystemExit(
            f"Field '{field_name}' must be a scalar in the input NPZ. "
            f"Got shape {array_value.shape}."
        )
    return array_value.reshape(()).item()


def validate_sequence_npz(npz_data) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int, int, float]:
    x = np.asarray(npz_data["X"])
    timestamps = np.asarray(npz_data["timestamps"])
    source_indices = np.asarray(npz_data["source_indices"])
    feature_names = decode_feature_names(np.asarray(npz_data["feature_names"]))
    seq_len = int(extract_scalar(npz_data["seq_len"], "seq_len"))
    stride = int(extract_scalar(npz_data["stride"], "stride"))
    max_time_gap = float(extract_scalar(npz_data["max_time_gap"], "max_time_gap"))

    if x.ndim != 3:
        raise SystemExit(f"Input X must have shape [N, T, 18]. Got shape {x.shape}.")
    if x.shape[2] != 18:
        raise SystemExit(
            f"Input X must have feature_dim=18. Got shape {x.shape}."
        )
    if x.shape[0] <= 0:
        raise SystemExit("Input X has zero sequences. At least one sequence is required.")
    if x.shape[1] != seq_len:
        raise SystemExit(
            f"Input X second dimension {x.shape[1]} does not match seq_len={seq_len}."
        )
    if timestamps.shape != x.shape[:2]:
        raise SystemExit(
            "Field 'timestamps' must have shape [N, T] matching X. "
            f"Got {timestamps.shape} vs expected {x.shape[:2]}."
        )
    if source_indices.shape != x.shape[:2]:
        raise SystemExit(
            "Field 'source_indices' must have shape [N, T] matching X. "
            f"Got {source_indices.shape} vs expected {x.shape[:2]}."
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
    return (
        x.astype(np.float32, copy=False),
        timestamps,
        source_indices,
        feature_names,
        seq_len,
        stride,
        max_time_gap,
    )


def compute_feature_statistics(
    x: np.ndarray,
    feature_names: list[str],
    epsilon: float,
    clip_percentile_low: float,
    clip_percentile_high: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    flat_x = x.reshape(-1, x.shape[2]).astype(np.float64, copy=False)
    feature_dim = flat_x.shape[1]

    stats: dict[str, np.ndarray] = {
        "mean": np.empty(feature_dim, dtype=np.float64),
        "std": np.empty(feature_dim, dtype=np.float64),
        "std_safe": np.empty(feature_dim, dtype=np.float64),
        "min": np.empty(feature_dim, dtype=np.float64),
        "max": np.empty(feature_dim, dtype=np.float64),
        "median": np.empty(feature_dim, dtype=np.float64),
        "p1": np.empty(feature_dim, dtype=np.float64),
        "p99": np.empty(feature_dim, dtype=np.float64),
        "finite_count": np.empty(feature_dim, dtype=np.int64),
        "nan_count": np.empty(feature_dim, dtype=np.int64),
        "inf_count": np.empty(feature_dim, dtype=np.int64),
        "clip_value_low": np.empty(feature_dim, dtype=np.float64),
        "clip_value_high": np.empty(feature_dim, dtype=np.float64),
    }

    for feature_index, feature_name in enumerate(feature_names):
        column = flat_x[:, feature_index]
        finite_mask = np.isfinite(column)
        finite_values = column[finite_mask]
        nan_count = int(np.isnan(column).sum())
        inf_count = int(np.isinf(column).sum())
        finite_count = int(finite_values.size)

        if finite_count == 0:
            raise SystemExit(
                f"Feature '{feature_name}' has zero finite values. "
                "Cannot compute normalization statistics."
            )

        std_value = float(np.std(finite_values, ddof=0))
        stats["mean"][feature_index] = float(np.mean(finite_values))
        stats["std"][feature_index] = std_value
        stats["std_safe"][feature_index] = 1.0 if std_value < epsilon else std_value
        stats["min"][feature_index] = float(np.min(finite_values))
        stats["max"][feature_index] = float(np.max(finite_values))
        stats["median"][feature_index] = float(np.median(finite_values))
        stats["p1"][feature_index] = float(np.percentile(finite_values, 1.0))
        stats["p99"][feature_index] = float(np.percentile(finite_values, 99.0))
        stats["finite_count"][feature_index] = finite_count
        stats["nan_count"][feature_index] = nan_count
        stats["inf_count"][feature_index] = inf_count
        stats["clip_value_low"][feature_index] = float(
            np.percentile(finite_values, clip_percentile_low)
        )
        stats["clip_value_high"][feature_index] = float(
            np.percentile(finite_values, clip_percentile_high)
        )

    std_lt_epsilon_indices = np.where(stats["std"] < epsilon)[0]
    return stats, std_lt_epsilon_indices.astype(np.int64, copy=False)


def build_normalized_x(x: np.ndarray, mean: np.ndarray, std_safe: np.ndarray) -> np.ndarray:
    mean_broadcast = mean.reshape(1, 1, -1)
    std_safe_broadcast = std_safe.reshape(1, 1, -1)
    normalized = (x.astype(np.float64, copy=False) - mean_broadcast) / std_safe_broadcast
    return normalized.astype(np.float32, copy=False)


def compute_normalized_summary(x_normalized: np.ndarray) -> dict[str, np.ndarray]:
    flat_x = x_normalized.reshape(-1, x_normalized.shape[2]).astype(np.float64, copy=False)
    feature_mean = np.empty(flat_x.shape[1], dtype=np.float64)
    feature_std = np.empty(flat_x.shape[1], dtype=np.float64)
    for feature_index in range(flat_x.shape[1]):
        finite_values = flat_x[:, feature_index][np.isfinite(flat_x[:, feature_index])]
        if finite_values.size == 0:
            feature_mean[feature_index] = np.nan
            feature_std[feature_index] = np.nan
            continue
        feature_mean[feature_index] = float(np.mean(finite_values))
        feature_std[feature_index] = float(np.std(finite_values, ddof=0))
    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }


def build_report_lines(
    args: argparse.Namespace,
    x: np.ndarray,
    feature_names: list[str],
    seq_len: int,
    stats: dict[str, np.ndarray],
    std_lt_epsilon_indices: np.ndarray,
    global_nan_count: int,
    global_inf_count: int,
    normalized_summary: dict[str, np.ndarray] | None,
) -> list[str]:
    num_sequences = int(x.shape[0])
    feature_dim = int(x.shape[2])

    lines = [
        "Mamba Pose Feature Normalization Report",
        "=" * 39,
        "",
        "Input / Output",
        f"- input_npz: {args.input}",
        f"- output_norm_npz: {args.output_norm}",
        (
            f"- output_normalized_npz: {args.output_normalized}"
            if args.save_normalized
            else "- output_normalized_npz: disabled"
        ),
        f"- output_report: {args.output_report}",
        "",
        "Sequence Summary",
        f"- X shape: {list(x.shape)}",
        f"- num_sequences: {num_sequences}",
        f"- seq_len: {seq_len}",
        f"- feature_dim: {feature_dim}",
        "",
        "Normalization Parameters",
        f"- epsilon: {format_scalar(args.epsilon)}",
        f"- clip_percentile_low: {format_scalar(args.clip_percentile_low)}",
        f"- clip_percentile_high: {format_scalar(args.clip_percentile_high)}",
        "",
        "Global Finite Checks",
        f"- X_global_nan_count: {global_nan_count}",
        f"- X_global_inf_count: {global_inf_count}",
    ]

    if std_lt_epsilon_indices.size > 0:
        std_lt_epsilon_names = [feature_names[index] for index in std_lt_epsilon_indices.tolist()]
        lines.append(
            "- std_lt_epsilon_features: " + ", ".join(std_lt_epsilon_names)
        )
    else:
        lines.append("- std_lt_epsilon_features: none")

    lines.extend(
        [
            "",
            "Training Reminder",
            "- Later training and ONNX inference must use the same mean/std_safe parameters saved by this script.",
            "- effective_feature_num is usually much larger in scale than avg_residual and bias terms, so feature normalization is necessary before model training.",
            "- This step only processes X features. It does not generate or modify correction labels y.",
            "",
            "Per-Feature Statistics",
        ]
    )

    for feature_index, feature_name in enumerate(feature_names):
        lines.extend(
            [
                f"- {feature_name}:",
                (
                    f"  mean={format_scalar(stats['mean'][feature_index])}, "
                    f"std={format_scalar(stats['std'][feature_index])}, "
                    f"std_safe={format_scalar(stats['std_safe'][feature_index])}"
                ),
                (
                    f"  min={format_scalar(stats['min'][feature_index])}, "
                    f"max={format_scalar(stats['max'][feature_index])}, "
                    f"median={format_scalar(stats['median'][feature_index])}"
                ),
                (
                    f"  p1={format_scalar(stats['p1'][feature_index])}, "
                    f"p99={format_scalar(stats['p99'][feature_index])}"
                ),
                (
                    f"  clip_low@p{format_scalar(args.clip_percentile_low)}="
                    f"{format_scalar(stats['clip_value_low'][feature_index])}, "
                    f"clip_high@p{format_scalar(args.clip_percentile_high)}="
                    f"{format_scalar(stats['clip_value_high'][feature_index])}"
                ),
                (
                    f"  finite_count={format_scalar(stats['finite_count'][feature_index])}, "
                    f"nan_count={format_scalar(stats['nan_count'][feature_index])}, "
                    f"inf_count={format_scalar(stats['inf_count'][feature_index])}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "How To Use In Training",
            "- Load mean and std_safe from the norm NPZ and apply X_normalized = (X - mean) / std_safe with feature-wise broadcasting.",
            "- Keep this exact norm NPZ version together with the trained model so the same parameters can be reused during ONNX-side preprocessing later.",
        ]
    )

    if normalized_summary is not None:
        feature_mean = normalized_summary["feature_mean"]
        feature_std = normalized_summary["feature_std"]
        lines.extend(
            [
                "",
                "Normalized Output Summary",
                (
                    "- normalized_feature_mean_range: "
                    f"min={format_scalar(np.nanmin(feature_mean))}, "
                    f"max={format_scalar(np.nanmax(feature_mean))}"
                ),
                (
                    "- normalized_feature_std_range: "
                    f"min={format_scalar(np.nanmin(feature_std))}, "
                    f"max={format_scalar(np.nanmax(feature_std))}"
                ),
            ]
        )

    return lines


def main() -> None:
    args = parse_args()
    validate_args(args)
    ensure_input_exists(args.input)

    npz_data = load_npz(args.input)
    validate_npz_fields(npz_data)
    x, timestamps, source_indices, feature_names, seq_len, stride, max_time_gap = (
        validate_sequence_npz(npz_data)
    )

    stats, std_lt_epsilon_indices = compute_feature_statistics(
        x=x,
        feature_names=feature_names,
        epsilon=args.epsilon,
        clip_percentile_low=args.clip_percentile_low,
        clip_percentile_high=args.clip_percentile_high,
    )

    flat_x = x.reshape(-1, x.shape[2]).astype(np.float64, copy=False)
    global_nan_count = int(np.isnan(flat_x).sum())
    global_inf_count = int(np.isinf(flat_x).sum())

    normalized_summary = None
    x_normalized = None
    if args.save_normalized:
        x_normalized = build_normalized_x(
            x=x,
            mean=stats["mean"],
            std_safe=stats["std_safe"],
        )
        normalized_summary = compute_normalized_summary(x_normalized)

    report_lines = build_report_lines(
        args=args,
        x=x,
        feature_names=feature_names,
        seq_len=seq_len,
        stats=stats,
        std_lt_epsilon_indices=std_lt_epsilon_indices,
        global_nan_count=global_nan_count,
        global_inf_count=global_inf_count,
        normalized_summary=normalized_summary,
    )

    args.output_norm.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    if args.save_normalized:
        args.output_normalized.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        args.output_norm,
        feature_names=np.asarray(feature_names, dtype=object),
        mean=stats["mean"],
        std=stats["std"],
        std_safe=stats["std_safe"],
        min=stats["min"],
        max=stats["max"],
        median=stats["median"],
        p1=stats["p1"],
        p99=stats["p99"],
        finite_count=stats["finite_count"],
        nan_count=stats["nan_count"],
        inf_count=stats["inf_count"],
        clip_percentile_low=np.asarray(args.clip_percentile_low, dtype=np.float64),
        clip_percentile_high=np.asarray(args.clip_percentile_high, dtype=np.float64),
        clip_value_low=stats["clip_value_low"],
        clip_value_high=stats["clip_value_high"],
        epsilon=np.asarray(args.epsilon, dtype=np.float64),
        input_path=np.asarray(str(args.input), dtype=object),
        num_sequences=np.asarray(x.shape[0], dtype=np.int64),
        seq_len=np.asarray(seq_len, dtype=np.int64),
        feature_dim=np.asarray(x.shape[2], dtype=np.int64),
    )

    if args.save_normalized and x_normalized is not None:
        np.savez(
            args.output_normalized,
            X_normalized=x_normalized,
            timestamps=timestamps,
            source_indices=source_indices,
            feature_names=np.asarray(feature_names, dtype=object),
            seq_len=np.asarray(seq_len, dtype=np.int64),
            stride=np.asarray(stride, dtype=np.int64),
            max_time_gap=np.asarray(max_time_gap, dtype=np.float64),
            norm_path=np.asarray(str(args.output_norm), dtype=object),
        )

    args.output_report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
