#!/usr/bin/env python3
"""
Build a pseudo correction-label dataset from dynamic FAST-LIVO2 trajectory data.

This script is intentionally limited to offline data preparation. It does not
modify FAST-LIVO2 runtime code, PoseCompensator inference logic, ONNX backend
behavior, the safety layer, launch files, or YAML configuration.

The generated labels are pseudo labels, not ground truth. Each label is aligned
to the last frame of a T-step input sequence so the offline supervision target
matches the runtime "history -> current-state correction" usage.
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
DEFAULT_CSV = REPO_ROOT / "Log" / "mamba_pose_train_data_interpolated.csv"
DEFAULT_SEQUENCES = (
    REPO_ROOT / "Log" / "mamba_pose_sequences_T10_interpolated_normalized.npz"
)
DEFAULT_NORM = REPO_ROOT / "Log" / "mamba_pose_feature_norm_T10_interpolated.npz"
DEFAULT_OUTPUT = REPO_ROOT / "Log" / "mamba_pose_pseudo_label_dataset_T10_smooth.npz"
DEFAULT_REPORT = (
    REPO_ROOT / "Log" / "mamba_pose_pseudo_label_dataset_T10_smooth_report.txt"
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

POSE_COLUMNS = [
    "timestamp",
    "pos_x",
    "pos_y",
    "pos_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "rot_w",
]

REQUIRED_SEQUENCE_FIELDS = [
    "X_normalized",
    "timestamps",
    "source_indices",
    "feature_names",
    "seq_len",
    "stride",
    "max_time_gap",
    "norm_path",
]

REQUIRED_NORM_FIELDS = ["feature_names", "mean", "std_safe"]

LABEL_NAMES = [
    "d_roll",
    "d_pitch",
    "d_yaw",
    "d_tx",
    "d_ty",
    "d_tz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a pseudo correction-label dataset by smoothing a dynamic "
            "FAST-LIVO2 trajectory and comparing each sequence target frame "
            "against its smoothed reference pose."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Interpolated trajectory CSV path. Default: {DEFAULT_CSV}",
    )
    parser.add_argument(
        "--sequences",
        type=Path,
        default=DEFAULT_SEQUENCES,
        help=f"Normalized sequence NPZ path. Default: {DEFAULT_SEQUENCES}",
    )
    parser.add_argument(
        "--norm",
        type=Path,
        default=DEFAULT_NORM,
        help=f"Feature norm NPZ path. Default: {DEFAULT_NORM}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Pseudo-label dataset NPZ path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Pseudo-label report TXT path. Default: {DEFAULT_REPORT}",
    )
    parser.add_argument(
        "--smooth-window-sec",
        type=float,
        default=1.0,
        help="Symmetric smoothing window width in seconds. Default: 1.0",
    )
    parser.add_argument(
        "--max-time-gap",
        type=float,
        default=0.2,
        help="Segment split threshold in seconds. Default: 0.2",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=3,
        help="Minimum same-segment neighbors required for a valid pseudo label. Default: 3",
    )
    parser.add_argument(
        "--max-rot-label-rad",
        type=float,
        default=0.10,
        help="Per-axis rotation-label absolute limit in radians. Default: 0.10",
    )
    parser.add_argument(
        "--max-trans-label-m",
        type=float,
        default=0.20,
        help="Per-axis translation-label absolute limit in meters. Default: 0.20",
    )
    parser.add_argument(
        "--label-target",
        default="last",
        help="Sequence label target frame. Only 'last' is supported. Default: last",
    )
    parser.add_argument(
        "--label-type",
        default="pseudo_smooth_reference",
        help=(
            "Label type metadata. Only 'pseudo_smooth_reference' is supported. "
            "Default: pseudo_smooth_reference"
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.smooth_window_sec <= 0 or not math.isfinite(args.smooth_window_sec):
        raise SystemExit("--smooth-window-sec must be a positive finite float.")
    if args.max_time_gap <= 0 or not math.isfinite(args.max_time_gap):
        raise SystemExit("--max-time-gap must be a positive finite float.")
    if args.min_neighbors <= 0:
        raise SystemExit("--min-neighbors must be a positive integer.")
    if args.max_rot_label_rad <= 0 or not math.isfinite(args.max_rot_label_rad):
        raise SystemExit("--max-rot-label-rad must be a positive finite float.")
    if args.max_trans_label_m <= 0 or not math.isfinite(args.max_trans_label_m):
        raise SystemExit("--max-trans-label-m must be a positive finite float.")
    if args.label_target != "last":
        raise SystemExit(
            f"Unsupported --label-target: {args.label_target}. Only 'last' is supported."
        )
    if args.label_type != "pseudo_smooth_reference":
        raise SystemExit(
            f"Unsupported --label-type: {args.label_type}. "
            "Only 'pseudo_smooth_reference' is supported."
        )


def ensure_input_exists(input_path: Path, description: str) -> None:
    if not input_path.exists():
        raise SystemExit(f"{description} not found: {input_path}")
    if not input_path.is_file():
        raise SystemExit(f"{description} is not a file: {input_path}")


def load_csv(input_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(input_path)
    except Exception as exc:
        raise SystemExit(f"Failed to read CSV '{input_path}': {exc}") from exc


def load_npz(input_path: Path):
    try:
        return np.load(input_path, allow_pickle=True)
    except Exception as exc:
        raise SystemExit(f"Failed to load NPZ '{input_path}': {exc}") from exc


def extract_scalar(array_like, field_name: str):
    array_value = np.asarray(array_like)
    if array_value.size != 1:
        raise SystemExit(
            f"Field '{field_name}' must be scalar-like. Got shape {array_value.shape}."
        )
    return array_value.reshape(()).item()


def decode_string_array(array_like) -> list[str]:
    decoded: list[str] = []
    for value in np.asarray(array_like).tolist():
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


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


def validate_csv(df: pd.DataFrame) -> None:
    missing = [column for column in POSE_COLUMNS if column not in df.columns]
    if missing:
        raise SystemExit(
            "Input CSV is missing required pose columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )
    if len(df) < 2:
        raise SystemExit(
            f"Input CSV must contain at least 2 rows. Got {len(df)} rows."
        )


def prepare_sorted_csv(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    prepared = df.copy()
    prepared["csv_row_index"] = np.arange(len(prepared), dtype=np.int64)
    for column in POSE_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    invalid_mask = ~np.isfinite(
        prepared[POSE_COLUMNS].to_numpy(dtype=np.float64)
    ).all(axis=1)
    if invalid_mask.any():
        raise SystemExit(
            "Input CSV contains non-finite values in required pose columns. "
            f"Invalid row count: {int(invalid_mask.sum())}"
        )

    quats = prepared[["rot_x", "rot_y", "rot_z", "rot_w"]].to_numpy(dtype=np.float64)
    quat_norms = np.linalg.norm(quats, axis=1)
    bad_quats = ~np.isfinite(quat_norms) | (quat_norms <= 1e-12)
    if bad_quats.any():
        raise SystemExit(
            "Input CSV contains invalid quaternions with zero or non-finite norm. "
            f"Invalid row count: {int(bad_quats.sum())}"
        )

    sorted_df = prepared.sort_values(
        by=["timestamp", "csv_row_index"], kind="mergesort"
    ).reset_index(drop=True)
    original_to_sorted = np.full(len(prepared), -1, dtype=np.int64)
    original_indices = sorted_df["csv_row_index"].to_numpy(dtype=np.int64)
    original_to_sorted[original_indices] = np.arange(len(sorted_df), dtype=np.int64)
    return sorted_df, original_to_sorted


def validate_sequence_fields(npz_data) -> None:
    missing = [field for field in REQUIRED_SEQUENCE_FIELDS if field not in npz_data]
    if missing:
        raise SystemExit(
            "Input sequence NPZ is missing required fields:\n"
            + "\n".join(f"  - {field}" for field in missing)
        )


def validate_norm_fields(npz_data) -> None:
    missing = [field for field in REQUIRED_NORM_FIELDS if field not in npz_data]
    if missing:
        raise SystemExit(
            "Input norm NPZ is missing required fields:\n"
            + "\n".join(f"  - {field}" for field in missing)
        )


def validate_sequences(npz_data):
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
            f"Input X_normalized must have shape [N, T, 18]. Got {x_normalized.shape}."
        )
    if x_normalized.shape[0] <= 0:
        raise SystemExit("Input X_normalized has zero sequences.")
    if x_normalized.shape[2] != 18:
        raise SystemExit(
            f"Input X_normalized must have feature_dim=18. Got {x_normalized.shape}."
        )
    if x_normalized.shape[1] != seq_len:
        raise SystemExit(
            f"Input X_normalized T dimension {x_normalized.shape[1]} does not match seq_len={seq_len}."
        )
    if timestamps.shape != x_normalized.shape[:2]:
        raise SystemExit(
            f"Field 'timestamps' must match [N, T]. Got {timestamps.shape} vs {x_normalized.shape[:2]}."
        )
    if source_indices.shape != x_normalized.shape[:2]:
        raise SystemExit(
            f"Field 'source_indices' must match [N, T]. Got {source_indices.shape} vs {x_normalized.shape[:2]}."
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
        timestamps.astype(np.float64, copy=False),
        source_indices.astype(np.int64, copy=False),
        feature_names,
        seq_len,
        stride,
        max_time_gap,
        norm_path_recorded,
        x_nan_count,
        x_inf_count,
    )


def validate_norm(npz_data, feature_names: list[str]) -> None:
    norm_feature_names = decode_string_array(npz_data["feature_names"])
    if norm_feature_names != feature_names:
        raise SystemExit(
            "Norm NPZ feature_names do not match sequence NPZ feature_names.\n"
            f"Norm: {', '.join(norm_feature_names)}\n"
            f"Sequence: {', '.join(feature_names)}"
        )


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise SystemExit("Encountered an invalid quaternion with zero or non-finite norm.")
    return q / norm


def align_quaternion_hemisphere(quaternion: np.ndarray, reference: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(quaternion)
    ref = normalize_quaternion(reference)
    if float(np.dot(q, ref)) < 0.0:
        q = -q
    return q


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(quaternion)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = normalize_quaternion(left)
    x2, y2, z2, w2 = normalize_quaternion(right)
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quaternion_to_rpy(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion(quaternion)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def compute_sorted_gap_stats(sorted_timestamps: np.ndarray) -> dict[str, object]:
    diffs = np.diff(sorted_timestamps) if sorted_timestamps.size > 1 else np.array([], dtype=np.float64)
    return {
        "diffs": diffs,
        "non_positive_gap_count": int(np.sum(diffs <= 0)),
        "large_gap_count": 0,
    }


def build_segments(sorted_timestamps: np.ndarray, max_time_gap: float) -> tuple[list[tuple[int, int]], dict[str, object]]:
    diffs = np.diff(sorted_timestamps) if sorted_timestamps.size > 1 else np.array([], dtype=np.float64)
    non_positive_gap_count = int(np.sum(diffs <= 0))
    large_gap_count = int(np.sum(diffs > max_time_gap))

    if sorted_timestamps.size == 0:
        return [], {
            "segment_count": 0,
            "segment_lengths": np.array([], dtype=np.int64),
            "non_positive_gap_count": non_positive_gap_count,
            "large_gap_count": large_gap_count,
        }

    segments: list[tuple[int, int]] = []
    start = 0
    for diff_index, dt in enumerate(diffs):
        if dt <= 0 or dt > max_time_gap:
            end = diff_index + 1
            segments.append((start, end))
            start = end
    segments.append((start, len(sorted_timestamps)))
    segment_lengths = np.array([end - begin for begin, end in segments], dtype=np.int64)
    return segments, {
        "segment_count": len(segments),
        "segment_lengths": segment_lengths,
        "non_positive_gap_count": non_positive_gap_count,
        "large_gap_count": large_gap_count,
    }


def smooth_sorted_trajectory(
    sorted_df: pd.DataFrame,
    segments: list[tuple[int, int]],
    smooth_window_sec: float,
    min_neighbors: int,
) -> dict[str, np.ndarray]:
    row_count = len(sorted_df)
    half_window_sec = smooth_window_sec * 0.5

    timestamps = sorted_df["timestamp"].to_numpy(dtype=np.float64)
    positions = sorted_df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64)
    quaternions = sorted_df[["rot_x", "rot_y", "rot_z", "rot_w"]].to_numpy(dtype=np.float64)
    quaternions = np.stack([normalize_quaternion(q) for q in quaternions], axis=0)

    ref_pos = np.full((row_count, 3), np.nan, dtype=np.float64)
    ref_quat = np.full((row_count, 4), np.nan, dtype=np.float64)
    neighbor_counts = np.zeros(row_count, dtype=np.int64)
    valid_by_min_neighbors = np.zeros(row_count, dtype=bool)

    for start, end in segments:
        segment_timestamps = timestamps[start:end]
        segment_positions = positions[start:end]
        segment_quats = quaternions[start:end]

        for local_index, current_timestamp in enumerate(segment_timestamps):
            window_start = np.searchsorted(
                segment_timestamps, current_timestamp - half_window_sec, side="left"
            )
            window_end = np.searchsorted(
                segment_timestamps, current_timestamp + half_window_sec, side="right"
            )
            neighbor_count = int(window_end - window_start)
            global_index = start + local_index
            neighbor_counts[global_index] = neighbor_count

            if neighbor_count < min_neighbors:
                continue

            valid_by_min_neighbors[global_index] = True
            ref_pos[global_index] = segment_positions[window_start:window_end].mean(axis=0)

            reference_quaternion = segment_quats[local_index]
            window_quats = segment_quats[window_start:window_end].copy()
            aligned_quats = np.stack(
                [
                    align_quaternion_hemisphere(q, reference_quaternion)
                    for q in window_quats
                ],
                axis=0,
            )
            ref_quat[global_index] = normalize_quaternion(aligned_quats.mean(axis=0))

    return {
        "timestamps": timestamps,
        "est_pos": positions,
        "est_quat": quaternions,
        "ref_pos": ref_pos,
        "ref_quat": ref_quat,
        "neighbor_counts": neighbor_counts,
        "valid_by_min_neighbors": valid_by_min_neighbors,
    }


def compute_label(est_pos: np.ndarray, est_quat: np.ndarray, ref_pos: np.ndarray, ref_quat: np.ndarray) -> np.ndarray:
    d_t = ref_pos - est_pos
    d_r = quaternion_multiply(quaternion_conjugate(est_quat), ref_quat)
    d_rpy = quaternion_to_rpy(d_r)
    return np.concatenate([d_rpy, d_t], axis=0).astype(np.float64, copy=False)


def summarize_dimension_stats(y: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "min": np.min(y, axis=0),
        "max": np.max(y, axis=0),
        "mean": np.mean(y, axis=0),
        "std": np.std(y, axis=0, ddof=0),
        "abs_max": np.max(np.abs(y), axis=0),
        "p50": np.percentile(y, 50, axis=0),
        "p90": np.percentile(y, 90, axis=0),
        "p95": np.percentile(y, 95, axis=0),
        "p99": np.percentile(y, 99, axis=0),
    }


def build_report_lines(
    args: argparse.Namespace,
    csv_path: Path,
    sequences_path: Path,
    norm_path: Path,
    output_path: Path,
    report_path: Path,
    csv_rows: int,
    original_sequence_count: int,
    output_valid_sequence_count: int,
    dropped_sequence_count: int,
    segment_stats: dict[str, object],
    smoothing_result: dict[str, np.ndarray],
    alignment_stats: dict[str, int],
    sequence_drop_stats: dict[str, int],
    y_stats: dict[str, np.ndarray],
    y_shape: tuple[int, ...],
    feature_names: list[str],
) -> list[str]:
    segment_lengths = np.asarray(segment_stats["segment_lengths"], dtype=np.int64)
    if segment_lengths.size > 0:
        segment_length_min = int(segment_lengths.min())
        segment_length_max = int(segment_lengths.max())
        segment_length_mean = float(segment_lengths.mean())
    else:
        segment_length_min = 0
        segment_length_max = 0
        segment_length_mean = 0.0

    neighbor_counts = smoothing_result["neighbor_counts"]
    avg_neighbor_count = float(neighbor_counts.mean()) if neighbor_counts.size else 0.0
    min_neighbor_count = int(neighbor_counts.min()) if neighbor_counts.size else 0
    max_neighbor_count = int(neighbor_counts.max()) if neighbor_counts.size else 0
    dropped_by_min_neighbors = int((~smoothing_result["valid_by_min_neighbors"]).sum())
    kept_ratio = (
        output_valid_sequence_count / original_sequence_count
        if original_sequence_count > 0
        else 0.0
    )

    lines = [
        "Mamba Pose Pseudo Label Dataset Report",
        "=" * 40,
        "",
        "Input / Output",
        f"- csv_path: {csv_path}",
        f"- sequences_path: {sequences_path}",
        f"- norm_path: {norm_path}",
        f"- output_dataset_path: {output_path}",
        f"- report_path: {report_path}",
        "",
        "Data Scale",
        f"- csv_rows: {csv_rows}",
        f"- original_sequence_count: {original_sequence_count}",
        f"- output_valid_sequence_count: {output_valid_sequence_count}",
        f"- dropped_sequence_count: {dropped_sequence_count}",
        "",
        "Smoothing Parameters",
        f"- smooth_window_sec: {format_scalar(args.smooth_window_sec)}",
        f"- max_time_gap: {format_scalar(args.max_time_gap)}",
        f"- min_neighbors: {args.min_neighbors}",
        f"- label_target: {args.label_target}",
        f"- max_rot_label_rad: {format_scalar(args.max_rot_label_rad)}",
        f"- max_trans_label_m: {format_scalar(args.max_trans_label_m)}",
        "",
        "Segment Statistics",
        f"- segment_count: {segment_stats['segment_count']}",
        (
            "- segment_length_stats: "
            f"min={segment_length_min}, max={segment_length_max}, "
            f"mean={format_scalar(segment_length_mean)}"
        ),
        f"- non_positive_gap_count: {segment_stats['non_positive_gap_count']}",
        f"- large_gap_count: {segment_stats['large_gap_count']}",
        "",
        "Smoothing Statistics",
        f"- average_neighbor_count: {format_scalar(avg_neighbor_count)}",
        f"- min_neighbor_count: {min_neighbor_count}",
        f"- max_neighbor_count: {max_neighbor_count}",
        f"- dropped_by_min_neighbors: {dropped_by_min_neighbors}",
        "",
        "Alignment Checks",
        f"- source_index_out_of_range_count: {alignment_stats['source_index_out_of_range_count']}",
        f"- timestamp_mismatch_count: {alignment_stats['timestamp_mismatch_count']}",
        "",
        "Label Filtering Statistics",
        f"- dropped_by_invalid_reference_frame: {sequence_drop_stats['dropped_by_invalid_reference_frame']}",
        f"- dropped_by_nan_inf: {sequence_drop_stats['dropped_by_nan_inf']}",
        f"- dropped_by_rot_limit: {sequence_drop_stats['dropped_by_rot_limit']}",
        f"- dropped_by_trans_limit: {sequence_drop_stats['dropped_by_trans_limit']}",
        f"- kept_ratio: {kept_ratio:.2%}",
        "",
        "Saved Dataset",
        f"- y shape: {list(y_shape)}",
        f"- label_type: {args.label_type}",
        f"- feature_names: {', '.join(feature_names)}",
        f"- label_names: {', '.join(LABEL_NAMES)}",
    ]

    for dim_index, label_name in enumerate(LABEL_NAMES):
        lines.extend(
            [
                (
                    f"- {label_name}: "
                    f"min={format_scalar(y_stats['min'][dim_index])}, "
                    f"max={format_scalar(y_stats['max'][dim_index])}, "
                    f"mean={format_scalar(y_stats['mean'][dim_index])}, "
                    f"std={format_scalar(y_stats['std'][dim_index])}"
                ),
                (
                    f"  abs_max={format_scalar(y_stats['abs_max'][dim_index])}, "
                    f"p50={format_scalar(y_stats['p50'][dim_index])}, "
                    f"p90={format_scalar(y_stats['p90'][dim_index])}, "
                    f"p95={format_scalar(y_stats['p95'][dim_index])}, "
                    f"p99={format_scalar(y_stats['p99'][dim_index])}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "Label Definition",
            "- Labels are aligned to the last frame of each sequence.",
            "- Translation label formula: d_t = p_ref - p_est.",
            "- Rotation label formula: d_R = R_est^T * R_ref.",
            "- d_R is converted to [d_roll, d_pitch, d_yaw].",
            "",
            "Smoothing Definition",
            (
                "- smooth_window_sec defines a symmetric timestamp window centered at the target frame, "
                "for example 1.0 sec -> [t_i - 0.5, t_i + 0.5]."
            ),
            "- Position smoothing uses uniform averaging inside the same continuous segment.",
            "- Rotation smoothing uses hemisphere-aligned quaternion averaging followed by normalization.",
            "",
            "Warnings",
            "- label_type=pseudo_smooth_reference is a pseudo label, not ground truth.",
            "- The smoothed reference trajectory is only a first teacher signal, not a true pose target.",
            "- If the smoothing window is too large, real turning motion can be over-smoothed.",
            "- Compare pseudo-label training results against the static_zero baseline and raw FAST-LIVO2 behavior.",
        ]
    )
    return lines


def save_dataset(
    output_path: Path,
    x_normalized: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    source_indices: np.ndarray,
    target_timestamps: np.ndarray,
    target_indices: np.ndarray,
    feature_names: list[str],
    seq_len: int,
    stride: int,
    max_time_gap: float,
    norm_path: Path,
    args: argparse.Namespace,
    csv_path: Path,
    sequences_path: Path,
    ref_pos: np.ndarray,
    est_pos: np.ndarray,
    ref_quat: np.ndarray,
    est_quat: np.ndarray,
    y_abs_max_per_sample: np.ndarray,
    is_pseudo_label_valid: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X_normalized=x_normalized,
        y=y.astype(np.float32, copy=False),
        timestamps=timestamps,
        source_indices=source_indices,
        target_timestamps=target_timestamps,
        target_indices=target_indices,
        feature_names=np.asarray(feature_names, dtype=object),
        label_names=np.asarray(LABEL_NAMES, dtype=object),
        seq_len=np.int64(seq_len),
        stride=np.int64(stride),
        max_time_gap=np.float64(max_time_gap),
        norm_path=np.asarray(str(norm_path), dtype=object),
        label_type=np.asarray(args.label_type, dtype=object),
        smooth_window_sec=np.float64(args.smooth_window_sec),
        max_rot_label_rad=np.float64(args.max_rot_label_rad),
        max_trans_label_m=np.float64(args.max_trans_label_m),
        input_csv_path=np.asarray(str(csv_path), dtype=object),
        input_sequences_path=np.asarray(str(sequences_path), dtype=object),
        num_sequences=np.int64(x_normalized.shape[0]),
        feature_dim=np.int64(x_normalized.shape[2]),
        label_dim=np.int64(y.shape[1]),
        ref_pos=ref_pos.astype(np.float32, copy=False),
        est_pos=est_pos.astype(np.float32, copy=False),
        ref_quat_xyzw=ref_quat.astype(np.float32, copy=False),
        est_quat_xyzw=est_quat.astype(np.float32, copy=False),
        y_abs_max_per_sample=y_abs_max_per_sample.astype(np.float32, copy=False),
        is_pseudo_label_valid=is_pseudo_label_valid,
    )


def write_report(report_path: Path, lines: list[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    validate_args(args)

    csv_path = args.csv.expanduser().resolve()
    sequences_path = args.sequences.expanduser().resolve()
    norm_path = args.norm.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()

    ensure_input_exists(csv_path, "Input CSV")
    ensure_input_exists(sequences_path, "Input sequences NPZ")
    ensure_input_exists(norm_path, "Input norm NPZ")

    csv_df = load_csv(csv_path)
    validate_csv(csv_df)
    sorted_df, original_to_sorted = prepare_sorted_csv(csv_df)

    sequences_npz = load_npz(sequences_path)
    validate_sequence_fields(sequences_npz)
    (
        x_normalized,
        sequence_timestamps,
        sequence_source_indices,
        feature_names,
        seq_len,
        stride,
        sequences_max_time_gap,
        norm_path_recorded,
        _x_nan_count,
        _x_inf_count,
    ) = validate_sequences(sequences_npz)

    norm_npz = load_npz(norm_path)
    validate_norm_fields(norm_npz)
    validate_norm(norm_npz, feature_names)

    sorted_timestamps = sorted_df["timestamp"].to_numpy(dtype=np.float64)
    segments, segment_stats = build_segments(sorted_timestamps, args.max_time_gap)
    smoothing_result = smooth_sorted_trajectory(
        sorted_df=sorted_df,
        segments=segments,
        smooth_window_sec=args.smooth_window_sec,
        min_neighbors=args.min_neighbors,
    )

    num_sequences = int(x_normalized.shape[0])
    csv_row_count = len(csv_df)
    target_indices_all = sequence_source_indices[:, -1].astype(np.int64, copy=False)
    target_timestamps_all = sequence_timestamps[:, -1].astype(np.float64, copy=False)

    kept_x: list[np.ndarray] = []
    kept_y: list[np.ndarray] = []
    kept_timestamps: list[np.ndarray] = []
    kept_source_indices: list[np.ndarray] = []
    kept_target_timestamps: list[float] = []
    kept_target_indices: list[int] = []
    kept_ref_pos: list[np.ndarray] = []
    kept_est_pos: list[np.ndarray] = []
    kept_ref_quat: list[np.ndarray] = []
    kept_est_quat: list[np.ndarray] = []
    kept_y_abs_max_per_sample: list[float] = []
    kept_is_valid: list[bool] = []

    alignment_stats = {
        "source_index_out_of_range_count": 0,
        "timestamp_mismatch_count": 0,
    }
    sequence_drop_stats = {
        "dropped_by_invalid_reference_frame": 0,
        "dropped_by_nan_inf": 0,
        "dropped_by_rot_limit": 0,
        "dropped_by_trans_limit": 0,
    }

    for sequence_index in range(num_sequences):
        target_index = int(target_indices_all[sequence_index])
        target_timestamp = float(target_timestamps_all[sequence_index])

        if target_index < 0 or target_index >= csv_row_count:
            alignment_stats["source_index_out_of_range_count"] += 1
            continue

        sorted_position = int(original_to_sorted[target_index])
        if sorted_position < 0 or sorted_position >= len(sorted_df):
            alignment_stats["source_index_out_of_range_count"] += 1
            continue

        csv_timestamp = float(sorted_timestamps[sorted_position])
        if abs(csv_timestamp - target_timestamp) > 1e-3:
            alignment_stats["timestamp_mismatch_count"] += 1
            continue

        if not bool(smoothing_result["valid_by_min_neighbors"][sorted_position]):
            sequence_drop_stats["dropped_by_invalid_reference_frame"] += 1
            continue

        est_pos = smoothing_result["est_pos"][sorted_position]
        est_quat = smoothing_result["est_quat"][sorted_position]
        ref_pos = smoothing_result["ref_pos"][sorted_position]
        ref_quat = smoothing_result["ref_quat"][sorted_position]

        if not np.isfinite(ref_pos).all() or not np.isfinite(ref_quat).all():
            sequence_drop_stats["dropped_by_invalid_reference_frame"] += 1
            continue

        y = compute_label(
            est_pos=est_pos,
            est_quat=est_quat,
            ref_pos=ref_pos,
            ref_quat=ref_quat,
        )

        if not np.isfinite(y).all():
            sequence_drop_stats["dropped_by_nan_inf"] += 1
            continue

        if np.any(np.abs(y[:3]) > args.max_rot_label_rad):
            sequence_drop_stats["dropped_by_rot_limit"] += 1
            continue

        if np.any(np.abs(y[3:]) > args.max_trans_label_m):
            sequence_drop_stats["dropped_by_trans_limit"] += 1
            continue

        kept_x.append(x_normalized[sequence_index])
        kept_y.append(y.astype(np.float32, copy=False))
        kept_timestamps.append(sequence_timestamps[sequence_index])
        kept_source_indices.append(sequence_source_indices[sequence_index])
        kept_target_timestamps.append(target_timestamp)
        kept_target_indices.append(target_index)
        kept_ref_pos.append(ref_pos.astype(np.float32, copy=False))
        kept_est_pos.append(est_pos.astype(np.float32, copy=False))
        kept_ref_quat.append(ref_quat.astype(np.float32, copy=False))
        kept_est_quat.append(est_quat.astype(np.float32, copy=False))
        kept_y_abs_max_per_sample.append(float(np.max(np.abs(y))))
        kept_is_valid.append(True)

    valid_sequence_count = len(kept_x)
    if valid_sequence_count == 0:
        raise SystemExit(
            "No valid pseudo-labeled sequences were kept. "
            "Check the trajectory quality, smoothing window, or label thresholds."
        )

    kept_x_array = np.stack(kept_x, axis=0).astype(np.float32, copy=False)
    kept_y_array = np.stack(kept_y, axis=0).astype(np.float32, copy=False)
    kept_timestamps_array = np.stack(kept_timestamps, axis=0).astype(np.float64, copy=False)
    kept_source_indices_array = np.stack(kept_source_indices, axis=0).astype(np.int64, copy=False)
    kept_target_timestamps_array = np.asarray(kept_target_timestamps, dtype=np.float64)
    kept_target_indices_array = np.asarray(kept_target_indices, dtype=np.int64)
    kept_ref_pos_array = np.stack(kept_ref_pos, axis=0).astype(np.float32, copy=False)
    kept_est_pos_array = np.stack(kept_est_pos, axis=0).astype(np.float32, copy=False)
    kept_ref_quat_array = np.stack(kept_ref_quat, axis=0).astype(np.float32, copy=False)
    kept_est_quat_array = np.stack(kept_est_quat, axis=0).astype(np.float32, copy=False)
    kept_y_abs_max_array = np.asarray(kept_y_abs_max_per_sample, dtype=np.float32)
    kept_is_valid_array = np.asarray(kept_is_valid, dtype=bool)

    y_stats = summarize_dimension_stats(kept_y_array.astype(np.float64, copy=False))

    save_dataset(
        output_path=output_path,
        x_normalized=kept_x_array,
        y=kept_y_array,
        timestamps=kept_timestamps_array,
        source_indices=kept_source_indices_array,
        target_timestamps=kept_target_timestamps_array,
        target_indices=kept_target_indices_array,
        feature_names=feature_names,
        seq_len=seq_len,
        stride=stride,
        max_time_gap=sequences_max_time_gap,
        norm_path=norm_path,
        args=args,
        csv_path=csv_path,
        sequences_path=sequences_path,
        ref_pos=kept_ref_pos_array,
        est_pos=kept_est_pos_array,
        ref_quat=kept_ref_quat_array,
        est_quat=kept_est_quat_array,
        y_abs_max_per_sample=kept_y_abs_max_array,
        is_pseudo_label_valid=kept_is_valid_array,
    )

    report_lines = build_report_lines(
        args=args,
        csv_path=csv_path,
        sequences_path=sequences_path,
        norm_path=norm_path,
        output_path=output_path,
        report_path=report_path,
        csv_rows=csv_row_count,
        original_sequence_count=num_sequences,
        output_valid_sequence_count=valid_sequence_count,
        dropped_sequence_count=num_sequences - valid_sequence_count,
        segment_stats=segment_stats,
        smoothing_result=smoothing_result,
        alignment_stats=alignment_stats,
        sequence_drop_stats=sequence_drop_stats,
        y_stats=y_stats,
        y_shape=kept_y_array.shape,
        feature_names=feature_names,
    )
    write_report(report_path, report_lines)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
