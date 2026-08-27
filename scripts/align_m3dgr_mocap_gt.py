#!/usr/bin/env python3
"""Align M3DGR TUM Mocap poses to a FAST-LIVO2 export CSV.

The tool never promotes an unknown Mocap rigid-body definition to trusted
ground truth.  With ``--rigid-body-frame-status unconfirmed`` it writes an
``alignment_candidate`` artifact even when all numerical checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from mamba_pose_se3 import (
    average_transforms,
    canonical_transform_hash,
    estimate_time_offset,
    interpolate_poses,
    invert_transforms,
    matrix_to_list,
    normalize_quaternions,
    percentile_summary,
    poses_from_transforms,
    quaternion_angular_distance_rad,
    transform_from_pose,
    transforms_from_poses,
)


POSE_COLUMNS = [
    "pos_x",
    "pos_y",
    "pos_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "rot_w",
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
FRAME_CONVENTION_VERSION = "m3dgr_mocap_gtbody_to_fastlivo_imu_right_se3_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lio-csv", type=Path, required=True)
    parser.add_argument("--mocap-tum", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--time-offset-sec",
        default="auto",
        help="Fixed offset where gt_query_time = lio_time + offset, or 'auto'.",
    )
    parser.add_argument("--time-search-half-range-sec", type=float, default=0.2)
    parser.add_argument("--time-search-step-sec", type=float, default=0.001)
    parser.add_argument("--max-interpolation-bracket-sec", type=float, default=0.020)
    parser.add_argument("--max-lio-segment-gap-sec", type=float, default=0.2)
    parser.add_argument("--initial-alignment-duration-sec", type=float, default=2.0)
    parser.add_argument(
        "--gt-body-to-imu",
        type=float,
        nargs=7,
        metavar=("TX", "TY", "TZ", "QX", "QY", "QZ", "QW"),
        help=(
            "T_gtBody_imu as translation plus xyzw quaternion. It is the right "
            "factor in T_world_imu = T_world_gtBody @ T_gtBody_imu."
        ),
    )
    parser.add_argument(
        "--candidate-identity-extrinsic",
        action="store_true",
        help="Use identity only to exercise the pipeline; output can never be trusted GT.",
    )
    parser.add_argument(
        "--gt-body-to-imu-report",
        type=Path,
        help="Hand-eye JSON containing shared_extrinsic.matrix.",
    )
    parser.add_argument(
        "--rigid-body-frame-status",
        choices=("confirmed", "unconfirmed"),
        required=True,
        help="Whether the physical Mocap rigid-body definition has been confirmed.",
    )
    parser.add_argument(
        "--extrinsic-source",
        default="unspecified",
        help="Human-readable calibration/metadata provenance recorded in the report.",
    )
    parser.add_argument("--min-match-ratio", type=float, default=0.95)
    parser.add_argument("--max-gt-quaternion-norm-error", type=float, default=1.0e-6)
    parser.add_argument("--max-initial-translation-median-m", type=float, default=0.02)
    parser.add_argument("--max-initial-rotation-median-deg", type=float, default=0.5)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path_name in ("lio_csv", "mocap_tum"):
        path = getattr(args, path_name).expanduser()
        if not path.is_file():
            raise SystemExit(f"Input file not found: {path}")
    extrinsic_option_count = sum(
        (
            args.gt_body_to_imu is not None,
            args.candidate_identity_extrinsic,
            args.gt_body_to_imu_report is not None,
        )
    )
    if extrinsic_option_count != 1:
        raise SystemExit(
            "Specify exactly one of --gt-body-to-imu, --gt-body-to-imu-report, "
            "or --candidate-identity-extrinsic."
        )
    if args.rigid_body_frame_status == "confirmed" and args.candidate_identity_extrinsic:
        raise SystemExit(
            "A candidate identity extrinsic cannot be paired with a confirmed rigid-body frame."
        )
    positive_names = (
        "time_search_half_range_sec",
        "time_search_step_sec",
        "max_interpolation_bracket_sec",
        "max_lio_segment_gap_sec",
        "initial_alignment_duration_sec",
        "min_match_ratio",
        "max_gt_quaternion_norm_error",
        "max_initial_translation_median_m",
        "max_initial_rotation_median_deg",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive.")
    if args.min_match_ratio > 1.0:
        raise SystemExit("--min-match-ratio cannot exceed 1.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lio_csv(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, float]:
    frame = pd.read_csv(path)
    required = ["timestamp", *POSE_COLUMNS, *GT_COLUMNS]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise SystemExit(f"LIO CSV is missing columns: {', '.join(missing)}")
    numeric = frame[["timestamp", *POSE_COLUMNS]].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        invalid = int((~np.isfinite(values)).any(axis=1).sum())
        raise SystemExit(f"LIO CSV contains {invalid} non-finite pose rows.")
    timestamps = numeric["timestamp"].to_numpy(dtype=np.float64)
    if not np.all(np.diff(timestamps) > 0.0):
        raise SystemExit("LIO CSV timestamps must be strictly increasing; clean/split it first.")
    positions = numeric[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64)
    quaternions_raw = numeric[["rot_x", "rot_y", "rot_z", "rot_w"]].to_numpy(
        dtype=np.float64
    )
    quaternions, norms = normalize_quaternions(quaternions_raw)
    return frame, timestamps, positions, quaternions, float(np.max(np.abs(norms - 1.0)))


def load_tum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    rows: list[list[float]] = []
    malformed = 0
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 8:
                malformed += 1
                continue
            try:
                rows.append([float(value) for value in fields[:8]])
            except ValueError as exc:
                raise SystemExit(f"Non-numeric TUM row at line {line_number}: {exc}") from exc
    if malformed:
        raise SystemExit(f"Mocap TUM contains {malformed} malformed non-comment rows.")
    if len(rows) < 2:
        raise SystemExit("Mocap TUM must contain at least two poses.")
    values = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(values).all():
        raise SystemExit("Mocap TUM contains NaN or Inf.")
    timestamps = values[:, 0]
    diffs = np.diff(timestamps)
    if not np.all(diffs > 0.0):
        raise SystemExit(
            "Mocap timestamps must be strictly increasing; duplicate/reversed rows are rejected."
        )
    quaternions, norms = normalize_quaternions(values[:, 4:8])
    stats: dict[str, float | int] = {
        "row_count": len(values),
        "timestamp_first": float(timestamps[0]),
        "timestamp_last": float(timestamps[-1]),
        "duration_sec": float(timestamps[-1] - timestamps[0]),
        "non_monotonic_count": int(np.sum(diffs <= 0.0)),
        "quaternion_norm_error_max": float(np.max(np.abs(norms - 1.0))),
    }
    return timestamps, values[:, 1:4], quaternions, stats


def parse_extrinsic(args: argparse.Namespace) -> np.ndarray:
    if args.candidate_identity_extrinsic:
        return np.eye(4, dtype=np.float64)
    if args.gt_body_to_imu_report is not None:
        report_path = args.gt_body_to_imu_report.expanduser().resolve()
        if not report_path.is_file():
            raise SystemExit(f"Hand-eye report not found: {report_path}")
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        matrix = payload.get("shared_extrinsic", {}).get("matrix")
        transform = np.asarray(matrix, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise SystemExit("Hand-eye report has no finite shared_extrinsic.matrix [4,4].")
        if payload.get("status") not in (
            "handeye_consistent_unconfirmed",
            "handeye_confirmed",
        ):
            raise SystemExit(
                f"Hand-eye report status '{payload.get('status')}' is not usable."
            )
        return transform
    values = np.asarray(args.gt_body_to_imu, dtype=np.float64)
    if not np.isfinite(values).all():
        raise SystemExit("--gt-body-to-imu contains NaN or Inf.")
    return transform_from_pose(values[:3], values[3:])


def determine_time_offset(
    args: argparse.Namespace,
    lio_timestamps: np.ndarray,
    lio_positions: np.ndarray,
    lio_quaternions: np.ndarray,
    gt_timestamps: np.ndarray,
    gt_positions: np.ndarray,
    gt_quaternions: np.ndarray,
) -> tuple[float, dict[str, object]]:
    if args.time_offset_sec != "auto":
        try:
            offset = float(args.time_offset_sec)
        except ValueError as exc:
            raise SystemExit("--time-offset-sec must be 'auto' or a finite number.") from exc
        if not math.isfinite(offset):
            raise SystemExit("--time-offset-sec must be finite.")
        return offset, {"mode": "fixed", "offset_sec": offset}
    try:
        result = estimate_time_offset(
            lio_timestamps,
            lio_positions,
            lio_quaternions,
            gt_timestamps,
            gt_positions,
            gt_quaternions,
            search_half_range_sec=args.time_search_half_range_sec,
            step_sec=args.time_search_step_sec,
        )
    except ValueError as exc:
        raise SystemExit(
            f"Automatic time-offset estimation failed: {exc} Supply a reviewed fixed offset."
        ) from exc
    return float(result["offset_sec"]), {"mode": "auto_speed_correlation", **result}


def build_segment_ids(
    timestamps: np.ndarray, valid: np.ndarray, max_gap_sec: float
) -> np.ndarray:
    segment_ids = np.full(len(timestamps), -1, dtype=np.int64)
    current_segment = -1
    previous_valid_index: int | None = None
    for index, is_valid in enumerate(valid):
        if not is_valid:
            previous_valid_index = None
            continue
        if (
            previous_valid_index is None
            or timestamps[index] - timestamps[previous_valid_index] > max_gap_sec
        ):
            current_segment += 1
        segment_ids[index] = current_segment
        previous_valid_index = index
    return segment_ids


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def main() -> int:
    args = parse_args()
    validate_args(args)
    lio_path = args.lio_csv.expanduser().resolve()
    mocap_path = args.mocap_tum.expanduser().resolve()
    output_path = args.output_csv.expanduser().resolve()
    report_path = args.report_json.expanduser().resolve()

    frame, lio_time, lio_pos, lio_quat, lio_quat_error = load_lio_csv(lio_path)
    gt_time, gt_pos, gt_quat, gt_stats = load_tum(mocap_path)
    gt_quaternion_gate = (
        gt_stats["quaternion_norm_error_max"] < args.max_gt_quaternion_norm_error
    )
    if not gt_quaternion_gate:
        raise SystemExit(
            "Mocap quaternion norm gate failed: "
            f"max_error={gt_stats['quaternion_norm_error_max']:.9g}, "
            f"required < {args.max_gt_quaternion_norm_error:.9g}."
        )

    time_offset, time_report = determine_time_offset(
        args, lio_time, lio_pos, lio_quat, gt_time, gt_pos, gt_quat
    )
    query_time = lio_time + time_offset
    interp_pos, interp_quat, valid, bracket_gaps = interpolate_poses(
        gt_time,
        gt_pos,
        gt_quat,
        query_time,
        args.max_interpolation_bracket_sec,
    )
    valid_count = int(valid.sum())
    if valid_count < 3:
        raise SystemExit(
            f"Only {valid_count} LIO frames match Mocap; cannot determine world alignment."
        )

    gt_body_to_imu = parse_extrinsic(args)
    mocap_valid = transforms_from_poses(interp_pos[valid], interp_quat[valid])
    gt_body_to_imu_batch = np.repeat(gt_body_to_imu[None, :, :], valid_count, axis=0)
    unaligned_ref_valid = mocap_valid @ gt_body_to_imu_batch
    lio_valid = transforms_from_poses(lio_pos[valid], lio_quat[valid])

    first_valid_time = float(lio_time[valid][0])
    initial_valid_local = lio_time[valid] <= (
        first_valid_time + args.initial_alignment_duration_sec
    )
    if int(initial_valid_local.sum()) < 3:
        raise SystemExit("Fewer than three matched poses fall inside the initial alignment window.")
    world_align_candidates = (
        lio_valid[initial_valid_local]
        @ invert_transforms(unaligned_ref_valid[initial_valid_local])
    )
    world_align = average_transforms(world_align_candidates)
    ref_valid = np.einsum("ij,njk->nik", world_align, unaligned_ref_valid)
    ref_pos_valid, ref_quat_valid = poses_from_transforms(ref_valid)

    initial_est_pos = lio_pos[valid][initial_valid_local]
    initial_est_quat = lio_quat[valid][initial_valid_local]
    initial_ref_pos = ref_pos_valid[initial_valid_local]
    initial_ref_quat = ref_quat_valid[initial_valid_local]
    initial_translation_error = np.linalg.norm(initial_ref_pos - initial_est_pos, axis=1)
    initial_rotation_error_rad = quaternion_angular_distance_rad(
        initial_ref_quat, initial_est_quat
    )
    initial_translation_stats = percentile_summary(initial_translation_error)
    initial_rotation_deg_stats = percentile_summary(
        np.degrees(initial_rotation_error_rad)
    )

    match_ratio = valid_count / len(frame) if len(frame) else 0.0
    gates = {
        "gt_timestamp_monotonic": gt_stats["non_monotonic_count"] == 0,
        "gt_quaternion_norm_error_lt_limit": bool(gt_quaternion_gate),
        "match_ratio_ge_limit": match_ratio >= args.min_match_ratio,
        "initial_translation_median_lt_limit": (
            initial_translation_stats["median"]
            < args.max_initial_translation_median_m
        ),
        "initial_rotation_median_lt_limit": (
            initial_rotation_deg_stats["median"] < args.max_initial_rotation_median_deg
        ),
        "rigid_body_frame_confirmed": args.rigid_body_frame_status == "confirmed",
    }
    numeric_gates_passed = all(
        value for key, value in gates.items() if key != "rigid_body_frame_confirmed"
    )
    all_gates_passed = numeric_gates_passed and gates["rigid_body_frame_confirmed"]
    if all_gates_passed:
        status = "trusted_gt"
    elif not gates["rigid_body_frame_confirmed"]:
        status = "alignment_candidate"
    else:
        status = "alignment_rejected"

    extrinsic_hash = canonical_transform_hash(
        gt_body_to_imu, FRAME_CONVENTION_VERSION
    )
    ref_positions = np.full((len(frame), 3), np.nan, dtype=np.float64)
    ref_quaternions = np.full((len(frame), 4), np.nan, dtype=np.float64)
    ref_positions[valid] = ref_pos_valid
    ref_quaternions[valid] = ref_quat_valid
    segment_ids = build_segment_ids(lio_time, valid, args.max_lio_segment_gap_sec)

    for column_index, column in enumerate(GT_COLUMNS[:3]):
        frame[column] = ref_positions[:, column_index]
    for column_index, column in enumerate(GT_COLUMNS[3:]):
        frame[column] = ref_quaternions[:, column_index]
    frame["gt_valid"] = valid.astype(np.int8)
    frame["gt_match_bracket_sec"] = np.where(valid, bracket_gaps, np.nan)
    frame["gt_type"] = np.where(valid, "mocap", "none")
    frame["gt_status"] = np.where(valid, status, "unmatched")
    frame["run_id"] = args.run_id
    frame["segment_id"] = segment_ids
    frame["gt_time_offset_sec"] = time_offset
    frame["gt_frame_convention_version"] = FRAME_CONVENTION_VERSION
    frame["gt_extrinsic_sha256"] = extrinsic_hash

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, float_format="%.15g")

    report = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": status,
        "label_eligibility": "trusted_mocap" if all_gates_passed else "not_trusted",
        "frame_convention_version": FRAME_CONVENTION_VERSION,
        "transform_convention": (
            "T_A_B maps frame B into frame A; "
            "T_ref=T_worldAlign@T_mocap@T_gtBody_imu"
        ),
        "inputs": {
            "lio_csv": str(lio_path),
            "lio_csv_sha256": sha256_file(lio_path),
            "mocap_tum": str(mocap_path),
            "mocap_tum_sha256": sha256_file(mocap_path),
        },
        "outputs": {"aligned_csv": str(output_path), "report_json": str(report_path)},
        "mocap_validation": gt_stats,
        "lio_validation": {
            "row_count": len(frame),
            "timestamp_first": float(lio_time[0]),
            "timestamp_last": float(lio_time[-1]),
            "quaternion_norm_error_max": lio_quat_error,
        },
        "time_alignment": time_report,
        "interpolation": {
            "max_bracket_sec": args.max_interpolation_bracket_sec,
            "matched_count": valid_count,
            "unmatched_count": int((~valid).sum()),
            "match_ratio": match_ratio,
            "matched_bracket_sec": percentile_summary(bracket_gaps[valid]),
        },
        "extrinsic": {
            "name": "T_gtBody_imu",
            "matrix": matrix_to_list(gt_body_to_imu),
            "sha256": extrinsic_hash,
            "source": args.extrinsic_source,
            "rigid_body_frame_status": args.rigid_body_frame_status,
            "candidate_identity_used": args.candidate_identity_extrinsic,
            "handeye_report": (
                str(args.gt_body_to_imu_report.expanduser().resolve())
                if args.gt_body_to_imu_report is not None
                else None
            ),
        },
        "world_alignment": {
            "source_window": "first matched 2 seconds only",
            "duration_sec": args.initial_alignment_duration_sec,
            "sample_count": int(initial_valid_local.sum()),
            "matrix": matrix_to_list(world_align),
        },
        "initial_alignment_residual": {
            "translation_m": initial_translation_stats,
            "rotation_deg": initial_rotation_deg_stats,
        },
        "segments": {
            "count": int(segment_ids.max() + 1) if valid_count else 0,
            "max_lio_gap_sec": args.max_lio_segment_gap_sec,
        },
        "acceptance_limits": {
            "gt_quaternion_norm_error_strictly_lt": args.max_gt_quaternion_norm_error,
            "match_ratio_ge": args.min_match_ratio,
            "initial_translation_median_strictly_lt_m": args.max_initial_translation_median_m,
            "initial_rotation_median_strictly_lt_deg": args.max_initial_rotation_median_deg,
        },
        "gates": gates,
        "numeric_gates_passed": numeric_gates_passed,
        "all_trust_gates_passed": all_gates_passed,
        "warnings": ([] if all_gates_passed else [
            "This artifact is not eligible as trusted training GT.",
            "Confirm the Mocap rigid-body physical frame and pass all numerical gates.",
        ]),
    }
    report_path.write_text(
        json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"run_id={args.run_id} status={status} matched={valid_count}/{len(frame)} "
        f"({match_ratio:.2%}) time_offset_sec={time_offset:.6f}"
    )
    print(f"aligned_csv={output_path}")
    print(f"report_json={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
