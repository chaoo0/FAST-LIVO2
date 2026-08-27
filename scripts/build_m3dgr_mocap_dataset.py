#!/usr/bin/env python3
"""Build auditable M3DGR right-SE(3) correction datasets.

Formal mode enforces the frozen bag-level split and derives feature
normalization only from training sequences.  Pilot mode is limited to
Varying-illu02 and intentionally does not create train-derived normalization.
No sample is removed because its correction exceeds a runtime safety limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mamba_pose_se3 import (
    TWIST_NAMES,
    invert_transforms,
    poses_from_transforms,
    quaternion_angular_distance_rad,
    se3_exp,
    se3_logs,
    transforms_from_poses,
)


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
GT_POSE_COLUMNS = [
    "gt_pos_x",
    "gt_pos_y",
    "gt_pos_z",
    "gt_rot_x",
    "gt_rot_y",
    "gt_rot_z",
    "gt_rot_w",
]
EST_POSE_COLUMNS = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "rot_w"]

FROZEN_SPLITS = {
    "train": (
        "Dynamic01",
        "Occlusion01",
        "Varying-illu01",
        "Dark03",
        "Wheel-float01",
        "Sha-turn01",
    ),
    "val": ("Dynamic02", "Wheel-float02"),
    "test": ("Occlusion02", "Varying-illu02", "Dark04", "Sha-turn02"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("formal", "pilot"), default="formal")
    parser.add_argument("--pilot-sequence", default="Varying-illu02")
    parser.add_argument("--allow-alignment-candidate", action="store_true")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-time-gap-sec", type=float, default=0.2)
    parser.add_argument("--normalization-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--max-time-offset-spread-sec", type=float, default=0.010)
    return parser.parse_args()


def resolve_manifest_path(raw: str, manifest_path: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def load_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"Manifest not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("sequences"), dict):
        raise SystemExit("Manifest must contain a 'sequences' mapping.")
    return payload


def validate_frozen_split(sequences: dict[str, object]) -> None:
    expected_names = {name for names in FROZEN_SPLITS.values() for name in names}
    actual_names = set(sequences)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        raise SystemExit(
            f"Manifest does not match frozen split. missing={missing}, extra={extra}"
        )
    for split, names in FROZEN_SPLITS.items():
        for name in names:
            entry = sequences[name]
            if not isinstance(entry, dict) or entry.get("split") != split:
                raise SystemExit(
                    f"Frozen split mismatch for {name}: expected '{split}', "
                    f"found '{entry.get('split') if isinstance(entry, dict) else None}'."
                )


def load_alignment_report(path: Path, run_id: str, allow_candidate: bool) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"Alignment report missing for {run_id}: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("run_id") != run_id:
        raise SystemExit(
            f"Alignment report run_id mismatch for {run_id}: {report.get('run_id')}"
        )
    status = report.get("status")
    if status != "trusted_gt" and not (allow_candidate and status == "alignment_candidate"):
        raise SystemExit(
            f"{run_id} alignment status is '{status}', not trusted_gt. "
            "Candidate artifacts are allowed only for explicit pipeline debugging."
        )
    return report


def contiguous_windows(
    timestamps: np.ndarray,
    valid: np.ndarray,
    segment_ids: np.ndarray,
    seq_len: int,
    stride: int,
    max_time_gap_sec: float,
) -> list[np.ndarray]:
    windows: list[np.ndarray] = []
    for start in range(0, len(timestamps) - seq_len + 1, stride):
        indices = np.arange(start, start + seq_len, dtype=np.int64)
        if not valid[indices].all():
            continue
        if segment_ids[indices[0]] < 0 or not np.all(segment_ids[indices] == segment_ids[indices[0]]):
            continue
        diffs = np.diff(timestamps[indices])
        if np.any(diffs <= 0.0) or np.any(diffs > max_time_gap_sec):
            continue
        windows.append(indices)
    return windows


def dimension_audit(values: np.ndarray, names: list[str] | tuple[str, ...]) -> dict[str, object]:
    result: dict[str, object] = {
        "shape": list(values.shape),
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
        "dimensions": {},
    }
    for index, name in enumerate(names):
        column = values[:, index]
        result["dimensions"][name] = {
            "mean": float(np.mean(column)),
            "std": float(np.std(column)),
            "mae_from_zero": float(np.mean(np.abs(column))),
            "p95_abs": float(np.percentile(np.abs(column), 95)),
            "p99_abs": float(np.percentile(np.abs(column), 99)),
            "max_abs": float(np.max(np.abs(column))),
            "min": float(np.min(column)),
            "max": float(np.max(column)),
        }
    return result


def build_run_arrays(
    run_id: str,
    csv_path: Path,
    report: dict[str, object],
    seq_len: int,
    stride: int,
    max_time_gap_sec: float,
) -> dict[str, object]:
    frame = pd.read_csv(csv_path)
    required = [
        "timestamp",
        *BASIC18_FEATURE_NAMES,
        *GT_POSE_COLUMNS,
        "gt_valid",
        "gt_type",
        "gt_status",
        "run_id",
        "segment_id",
        "gt_time_offset_sec",
        "gt_frame_convention_version",
        "gt_extrinsic_sha256",
    ]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise SystemExit(f"{run_id} aligned CSV is missing: {', '.join(missing)}")
    if set(frame["run_id"].astype(str)) != {run_id}:
        raise SystemExit(f"{run_id} aligned CSV has inconsistent run_id values.")

    numeric_names = ["timestamp", *BASIC18_FEATURE_NAMES, *GT_POSE_COLUMNS, "segment_id"]
    numeric = frame[numeric_names].apply(pd.to_numeric, errors="coerce")
    timestamps = numeric["timestamp"].to_numpy(dtype=np.float64)
    if not np.all(np.diff(timestamps) > 0.0):
        raise SystemExit(f"{run_id} aligned CSV timestamps are not strictly increasing.")
    x_rows = numeric[BASIC18_FEATURE_NAMES].to_numpy(dtype=np.float64)
    est_pose_rows = numeric[EST_POSE_COLUMNS].to_numpy(dtype=np.float64)
    ref_pose_rows = numeric[GT_POSE_COLUMNS].to_numpy(dtype=np.float64)
    segment_ids = numeric["segment_id"].to_numpy(dtype=np.int64)
    gt_valid = pd.to_numeric(frame["gt_valid"], errors="coerce").fillna(0).to_numpy(dtype=np.int64) == 1
    gt_valid &= frame["gt_type"].astype(str).to_numpy() == "mocap"
    finite_rows = np.isfinite(x_rows).all(axis=1) & np.isfinite(est_pose_rows).all(axis=1)
    finite_gt = np.isfinite(ref_pose_rows).all(axis=1)
    usable = gt_valid & finite_rows & finite_gt
    windows = contiguous_windows(
        timestamps, usable, segment_ids, seq_len, stride, max_time_gap_sec
    )
    if not windows:
        raise SystemExit(f"{run_id} produced no valid length-{seq_len} windows.")
    source_indices = np.stack(windows)
    target_indices = source_indices[:, -1]
    x_raw = x_rows[source_indices].astype(np.float32)
    est_pose = est_pose_rows[target_indices]
    ref_pose = ref_pose_rows[target_indices]
    est_transforms = transforms_from_poses(est_pose[:, :3], est_pose[:, 3:])
    ref_transforms = transforms_from_poses(ref_pose[:, :3], ref_pose[:, 3:])
    delta = invert_transforms(est_transforms) @ ref_transforms
    labels = se3_logs(delta).astype(np.float32)
    if not np.isfinite(labels).all():
        raise SystemExit(f"{run_id} generated non-finite SE(3) labels.")

    reconstructed = np.stack(
        [est @ se3_exp(label) for est, label in zip(est_transforms, labels)], axis=0
    )
    reconstructed_pos, reconstructed_quat = poses_from_transforms(reconstructed)
    reconstruction_translation_error = np.linalg.norm(reconstructed_pos - ref_pose[:, :3], axis=1)
    reconstruction_rotation_error = quaternion_angular_distance_rad(
        reconstructed_quat, ref_pose[:, 3:]
    )
    max_reconstruction_error = max(
        float(np.max(reconstruction_translation_error)),
        float(np.max(reconstruction_rotation_error)),
    )
    if max_reconstruction_error > 1.0e-6:
        raise SystemExit(
            f"{run_id} Exp(Log(.)) reconstruction error is too large: {max_reconstruction_error}"
        )

    status_values = set(frame.loc[target_indices, "gt_status"].astype(str))
    report_status = str(report["status"])
    if status_values != {report_status}:
        raise SystemExit(
            f"{run_id} target gt_status values {status_values} disagree with report {report_status}."
        )
    time_offsets = pd.to_numeric(
        frame.loc[target_indices, "gt_time_offset_sec"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(time_offsets).all() or np.ptp(time_offsets) > 1.0e-12:
        raise SystemExit(f"{run_id} has inconsistent time-offset provenance.")
    extrinsic_hashes = set(frame.loc[target_indices, "gt_extrinsic_sha256"].astype(str))
    frame_versions = set(frame.loc[target_indices, "gt_frame_convention_version"].astype(str))
    if len(extrinsic_hashes) != 1 or len(frame_versions) != 1:
        raise SystemExit(f"{run_id} has inconsistent frame/extrinsic provenance.")
    return {
        "run_id": run_id,
        "X_raw": x_raw,
        "y": labels,
        "timestamps": timestamps[source_indices],
        "source_indices": source_indices,
        "target_indices": target_indices,
        "segment_ids": segment_ids[target_indices],
        "est_pose_xyzw": est_pose.astype(np.float64),
        "ref_pose_xyzw": ref_pose.astype(np.float64),
        "time_offset_sec": float(time_offsets[0]),
        "extrinsic_hash": next(iter(extrinsic_hashes)),
        "frame_version": next(iter(frame_versions)),
        "alignment_status": report_status,
        "reconstruction_translation_error_max_m": float(np.max(reconstruction_translation_error)),
        "reconstruction_rotation_error_max_rad": float(np.max(reconstruction_rotation_error)),
        "csv_row_count": len(frame),
        "usable_row_count": int(usable.sum()),
        "window_count": len(windows),
    }


def concatenate_runs(run_arrays: list[dict[str, object]]) -> dict[str, np.ndarray]:
    fields = (
        "X_raw",
        "y",
        "timestamps",
        "source_indices",
        "target_indices",
        "segment_ids",
        "est_pose_xyzw",
        "ref_pose_xyzw",
    )
    combined = {field: np.concatenate([run[field] for run in run_arrays], axis=0) for field in fields}
    combined["run_ids"] = np.concatenate(
        [np.full(run["window_count"], run["run_id"], dtype="U64") for run in run_arrays]
    )
    combined["time_offset_sec"] = np.concatenate(
        [np.full(run["window_count"], run["time_offset_sec"], dtype=np.float64) for run in run_arrays]
    )
    combined["extrinsic_hashes"] = np.concatenate(
        [np.full(run["window_count"], run["extrinsic_hash"], dtype="U64") for run in run_arrays]
    )
    return combined


def save_split(
    path: Path,
    arrays: dict[str, np.ndarray],
    split: str,
    normalized: np.ndarray | None,
    norm_path: Path | None,
    seq_len: int,
    stride: int,
    label_type: str,
    frame_version: str,
) -> None:
    payload = {
        **arrays,
        "feature_names": np.asarray(BASIC18_FEATURE_NAMES, dtype="U64"),
        "label_names": np.asarray(TWIST_NAMES, dtype="U64"),
        "seq_len": np.int64(seq_len),
        "stride": np.int64(stride),
        "split": np.asarray(split),
        "gt_type": np.asarray("mocap"),
        "label_type": np.asarray(label_type),
        "frame_convention_version": np.asarray(frame_version),
        "runtime_safety_filter_applied": np.bool_(False),
    }
    if normalized is not None:
        payload["X_normalized"] = normalized.astype(np.float32)
        payload["norm_path"] = np.asarray(str(norm_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def main() -> int:
    args = parse_args()
    if args.seq_len <= 0 or args.stride <= 0 or args.max_time_gap_sec <= 0:
        raise SystemExit("Sequence length, stride, and max time gap must be positive.")
    if args.mode == "formal" and args.allow_alignment_candidate:
        raise SystemExit("Formal dataset construction never accepts alignment candidates.")
    if args.mode == "pilot" and args.pilot_sequence != "Varying-illu02":
        raise SystemExit("The frozen Pilot sequence is Varying-illu02; it cannot be replaced silently.")

    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    sequences = manifest["sequences"]
    validate_frozen_split(sequences)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_names = (
        [args.pilot_sequence]
        if args.mode == "pilot"
        else [name for split in ("train", "val", "test") for name in FROZEN_SPLITS[split]]
    )
    run_arrays: dict[str, dict[str, object]] = {}
    reports: dict[str, dict[str, object]] = {}
    missing_artifacts: list[str] = []
    for run_id in selected_names:
        entry = sequences[run_id]
        csv_raw = str(entry.get("aligned_csv", "")).strip()
        report_raw = str(entry.get("alignment_report", "")).strip()
        if not csv_raw or not report_raw:
            missing_artifacts.append(
                f"{run_id}: aligned_csv/alignment_report path is not configured"
            )
            continue
        csv_path = resolve_manifest_path(csv_raw, manifest_path)
        report_path = resolve_manifest_path(report_raw, manifest_path)
        if not csv_path.is_file() or not report_path.is_file():
            missing_artifacts.append(
                f"{run_id}: aligned_csv={csv_path} alignment_report={report_path}"
            )
            continue
        report = load_alignment_report(
            report_path, run_id, args.allow_alignment_candidate and args.mode == "pilot"
        )
        reports[run_id] = report
        run_arrays[run_id] = build_run_arrays(
            run_id,
            csv_path,
            report,
            args.seq_len,
            args.stride,
            args.max_time_gap_sec,
        )
    if missing_artifacts:
        raise SystemExit(
            "Required M3DGR artifacts are missing; the frozen split was not modified:\n  - "
            + "\n  - ".join(missing_artifacts)
        )

    statuses = {run["alignment_status"] for run in run_arrays.values()}
    label_type = (
        "mocap_right_se3_twist"
        if statuses == {"trusted_gt"}
        else "alignment_candidate_right_se3_twist"
    )
    frame_versions = {run["frame_version"] for run in run_arrays.values()}
    extrinsic_hashes = {run["extrinsic_hash"] for run in run_arrays.values()}
    if len(frame_versions) != 1:
        raise SystemExit(f"Frame convention differs across runs: {sorted(frame_versions)}")
    if args.mode == "formal" and len(extrinsic_hashes) != 1:
        raise SystemExit(
            "Formal runs do not share one Mocap-to-IMU extrinsic hash; stop label generation."
        )

    report_payload: dict[str, object] = {
        "schema_version": 1,
        "mode": args.mode,
        "label_type": label_type,
        "gt_type": "mocap",
        "frame_convention_version": next(iter(frame_versions)),
        "seq_len": args.seq_len,
        "stride": args.stride,
        "max_time_gap_sec": args.max_time_gap_sec,
        "runtime_safety_filter_applied": False,
        "runs": {},
        "outputs": {},
    }
    for run_id, run in run_arrays.items():
        report_payload["runs"][run_id] = {
            key: value
            for key, value in run.items()
            if key
            not in {
                "X_raw",
                "y",
                "timestamps",
                "source_indices",
                "target_indices",
                "segment_ids",
                "est_pose_xyzw",
                "ref_pose_xyzw",
            }
        }

    if args.mode == "pilot":
        combined = concatenate_runs([run_arrays[args.pilot_sequence]])
        output_path = output_dir / f"m3dgr_{args.pilot_sequence}_pilot_T{args.seq_len}.npz"
        save_split(
            output_path,
            combined,
            "pilot",
            None,
            None,
            args.seq_len,
            args.stride,
            label_type,
            next(iter(frame_versions)),
        )
        report_payload["outputs"]["pilot_dataset"] = str(output_path)
        report_payload["label_audit"] = dimension_audit(combined["y"], TWIST_NAMES)
        report_payload["normalization"] = {
            "created": False,
            "reason": "Pilot is a frozen test sequence; training statistics must not be derived from it.",
        }
    else:
        combined_by_split: dict[str, dict[str, np.ndarray]] = {}
        for split, names in FROZEN_SPLITS.items():
            combined_by_split[split] = concatenate_runs([run_arrays[name] for name in names])
        run_sets = {
            split: set(arrays["run_ids"].tolist()) for split, arrays in combined_by_split.items()
        }
        if run_sets["train"] & run_sets["val"] or run_sets["train"] & run_sets["test"] or run_sets["val"] & run_sets["test"]:
            raise SystemExit("Data isolation failure: run IDs overlap across splits.")
        time_offsets = np.asarray(
            [run_arrays[name]["time_offset_sec"] for name in selected_names], dtype=np.float64
        )
        time_offset_spread = float(np.ptp(time_offsets))
        if time_offset_spread > args.max_time_offset_spread_sec:
            raise SystemExit(
                f"Time-offset spread {time_offset_spread:.6f}s exceeds "
                f"{args.max_time_offset_spread_sec:.6f}s; inspect acquisition synchronization."
            )
        train_flat = combined_by_split["train"]["X_raw"].reshape(-1, len(BASIC18_FEATURE_NAMES)).astype(np.float64)
        mean = train_flat.mean(axis=0)
        std = train_flat.std(axis=0)
        std_safe = np.where(std < args.normalization_epsilon, args.normalization_epsilon, std)
        norm_path = output_dir / f"m3dgr_feature_norm_train_T{args.seq_len}.npz"
        np.savez(
            norm_path,
            feature_names=np.asarray(BASIC18_FEATURE_NAMES, dtype="U64"),
            mean=mean,
            std=std,
            std_safe=std_safe,
            epsilon=np.float64(args.normalization_epsilon),
            source_split=np.asarray("train"),
            source_run_ids=np.asarray(FROZEN_SPLITS["train"], dtype="U64"),
            train_frame_value_count=np.int64(len(train_flat)),
            train_raw_sha256=np.asarray(sha256_array(train_flat)),
        )
        report_payload["normalization"] = {
            "created": True,
            "path": str(norm_path),
            "source_split": "train",
            "source_run_ids": list(FROZEN_SPLITS["train"]),
            "train_frame_value_count": len(train_flat),
            "std_lt_epsilon_features": [
                BASIC18_FEATURE_NAMES[index] for index in np.flatnonzero(std < args.normalization_epsilon)
            ],
        }
        report_payload["time_offset_spread_sec"] = time_offset_spread
        report_payload["data_isolation"] = {
            "passed": True,
            "run_ids": {split: sorted(values) for split, values in run_sets.items()},
        }
        report_payload["label_audit"] = {}
        for split, arrays in combined_by_split.items():
            normalized = ((arrays["X_raw"].astype(np.float64) - mean) / std_safe).astype(np.float32)
            output_path = output_dir / f"m3dgr_mocap_{split}_T{args.seq_len}.npz"
            save_split(
                output_path,
                arrays,
                split,
                normalized,
                norm_path,
                args.seq_len,
                args.stride,
                label_type,
                next(iter(frame_versions)),
            )
            report_payload["outputs"][f"{split}_dataset"] = str(output_path)
            report_payload["label_audit"][split] = dimension_audit(arrays["y"], TWIST_NAMES)

    report_path = output_dir / f"m3dgr_mocap_dataset_T{args.seq_len}_report.json"
    report_path.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"mode={args.mode} label_type={label_type} runs={len(run_arrays)} "
        f"report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
