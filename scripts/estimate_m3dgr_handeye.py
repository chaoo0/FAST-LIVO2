#!/usr/bin/env python3
"""Estimate a shared M3DGR Mocap-body to MID360-IMU hand-eye transform.

Input is a YAML manifest with at least two runs::

    runs:
      - run_id: Dynamic01
        lio_csv: /path/to/mamba_pose_train_data_clean.csv
        mocap_tum: /path/to/Dynamic01.txt
        time_offset_sec: 0.012

The solved equation is ``A X = X B`` with Mocap relative motion ``A``, LIO
relative motion ``B``, and ``X = T_gtBody_imu``.  A numerically consistent
result remains ``handeye_consistent_unconfirmed`` until the physical rigid-body
definition is confirmed with dataset metadata or the dataset authors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from align_m3dgr_mocap_gt import load_lio_csv, load_tum
from mamba_pose_se3 import (
    canonical_transform_hash,
    interpolate_poses,
    invert_transform,
    matrix_to_list,
    se3_exp,
    se3_log,
    transforms_from_poses,
)


FRAME_CONVENTION_VERSION = "m3dgr_mocap_gtbody_to_fastlivo_imu_right_se3_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--relative-interval-sec", type=float, default=0.5)
    parser.add_argument("--relative-stride", type=int, default=5)
    parser.add_argument("--max-interpolation-bracket-sec", type=float, default=0.020)
    parser.add_argument("--min-rotation-deg", type=float, default=1.0)
    parser.add_argument("--min-relative-motions-per-run", type=int, default=20)
    parser.add_argument("--max-cross-run-translation-m", type=float, default=0.02)
    parser.add_argument("--max-cross-run-rotation-deg", type=float, default=1.0)
    parser.add_argument("--max-nfev", type=int, default=500)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise SystemExit(f"Manifest not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    runs = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(runs, list) or len(runs) < 2:
        raise SystemExit("Hand-eye manifest must contain at least two entries under 'runs'.")
    required = {"run_id", "lio_csv", "mocap_tum", "time_offset_sec"}
    for index, run in enumerate(runs):
        if not isinstance(run, dict) or not required.issubset(run):
            raise SystemExit(f"Manifest run {index} must define: {', '.join(sorted(required))}")
    return runs


def select_relative_motions(
    timestamps: np.ndarray,
    mocap_transforms: np.ndarray,
    lio_transforms: np.ndarray,
    interval_sec: float,
    stride: int,
    min_rotation_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    motion_a: list[np.ndarray] = []
    motion_b: list[np.ndarray] = []
    for first_index in range(0, len(timestamps), stride):
        second_index = int(np.searchsorted(timestamps, timestamps[first_index] + interval_sec))
        if second_index >= len(timestamps):
            continue
        relative_mocap = invert_transform(mocap_transforms[first_index]) @ mocap_transforms[second_index]
        relative_lio = invert_transform(lio_transforms[first_index]) @ lio_transforms[second_index]
        rotation_excitation = max(
            np.linalg.norm(Rotation.from_matrix(relative_mocap[:3, :3]).as_rotvec()),
            np.linalg.norm(Rotation.from_matrix(relative_lio[:3, :3]).as_rotvec()),
        )
        if rotation_excitation < min_rotation_rad:
            continue
        motion_a.append(relative_mocap)
        motion_b.append(relative_lio)
    if not motion_a:
        return np.empty((0, 4, 4)), np.empty((0, 4, 4))
    return np.stack(motion_a), np.stack(motion_b)


def solve_handeye(motion_a: np.ndarray, motion_b: np.ndarray, max_nfev: int) -> dict[str, object]:
    if len(motion_a) != len(motion_b) or not len(motion_a):
        raise ValueError("Hand-eye solver requires paired relative motions.")

    def residual(parameters: np.ndarray) -> np.ndarray:
        transform = se3_exp(parameters)
        errors = []
        for first, second in zip(motion_a, motion_b):
            lhs = first @ transform
            rhs = transform @ second
            errors.append(se3_log(invert_transform(lhs) @ rhs))
        return np.concatenate(errors)

    result = least_squares(
        residual,
        np.zeros(6, dtype=np.float64),
        loss="soft_l1",
        f_scale=0.02,
        max_nfev=max_nfev,
    )
    transform = se3_exp(result.x)
    residual_matrix = residual(result.x).reshape(-1, 6)
    return {
        "transform": transform,
        "success": bool(result.success),
        "message": result.message,
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "motion_count": len(motion_a),
        "rotation_residual_deg_median": float(
            np.median(np.linalg.norm(residual_matrix[:, :3], axis=1)) * 180.0 / np.pi
        ),
        "translation_residual_median_m": float(
            np.median(np.linalg.norm(residual_matrix[:, 3:], axis=1))
        ),
    }


def main() -> int:
    args = parse_args()
    if args.relative_interval_sec <= 0 or args.relative_stride <= 0:
        raise SystemExit("Relative interval and stride must be positive.")
    runs = load_manifest(args.manifest.expanduser().resolve())
    per_run: list[dict[str, object]] = []
    combined_a: list[np.ndarray] = []
    combined_b: list[np.ndarray] = []

    for run in runs:
        run_id = str(run["run_id"])
        lio_path = Path(str(run["lio_csv"])).expanduser().resolve()
        mocap_path = Path(str(run["mocap_tum"])).expanduser().resolve()
        offset = float(run["time_offset_sec"])
        _, lio_time, lio_pos, lio_quat, _ = load_lio_csv(lio_path)
        gt_time, gt_pos, gt_quat, _ = load_tum(mocap_path)
        interp_pos, interp_quat, valid, _ = interpolate_poses(
            gt_time,
            gt_pos,
            gt_quat,
            lio_time + offset,
            args.max_interpolation_bracket_sec,
        )
        timestamps = lio_time[valid]
        mocap_transforms = transforms_from_poses(interp_pos[valid], interp_quat[valid])
        lio_transforms = transforms_from_poses(lio_pos[valid], lio_quat[valid])
        motion_a, motion_b = select_relative_motions(
            timestamps,
            mocap_transforms,
            lio_transforms,
            args.relative_interval_sec,
            args.relative_stride,
            np.deg2rad(args.min_rotation_deg),
        )
        if len(motion_a) < args.min_relative_motions_per_run:
            raise SystemExit(
                f"{run_id} has only {len(motion_a)} observable relative motions; "
                f"need at least {args.min_relative_motions_per_run}."
            )
        solution = solve_handeye(motion_a, motion_b, args.max_nfev)
        per_run.append({"run_id": run_id, **solution})
        combined_a.append(motion_a)
        combined_b.append(motion_b)

    shared = solve_handeye(np.concatenate(combined_a), np.concatenate(combined_b), args.max_nfev)
    run_transforms = [entry["transform"] for entry in per_run]
    pairwise: list[dict[str, float | str]] = []
    for first_index in range(len(per_run)):
        for second_index in range(first_index + 1, len(per_run)):
            delta = invert_transform(run_transforms[first_index]) @ run_transforms[second_index]
            pairwise.append(
                {
                    "first": str(per_run[first_index]["run_id"]),
                    "second": str(per_run[second_index]["run_id"]),
                    "translation_difference_m": float(np.linalg.norm(delta[:3, 3])),
                    "rotation_difference_deg": float(
                        np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec())
                        * 180.0
                        / np.pi
                    ),
                }
            )
    max_translation = max(item["translation_difference_m"] for item in pairwise)
    max_rotation = max(item["rotation_difference_deg"] for item in pairwise)
    cross_run_consistent = (
        max_translation < args.max_cross_run_translation_m
        and max_rotation < args.max_cross_run_rotation_deg
    )
    status = "handeye_consistent_unconfirmed" if cross_run_consistent else "handeye_rejected"

    for entry in per_run:
        transform = entry.pop("transform")
        entry["matrix"] = matrix_to_list(transform)
    shared_transform = shared.pop("transform")
    payload = {
        "schema_version": 1,
        "status": status,
        "trusted_gt_eligible": False,
        "reason": (
            "Hand-eye is numerically consistent but physical rigid-body metadata remains unconfirmed."
            if cross_run_consistent
            else "Per-run hand-eye solutions exceed the 2 cm / 1 degree consistency gate."
        ),
        "equation": "A_mocap_relative @ X = X @ B_lio_relative",
        "frame_convention_version": FRAME_CONVENTION_VERSION,
        "shared_extrinsic": {
            **shared,
            "matrix": matrix_to_list(shared_transform),
            "sha256": canonical_transform_hash(shared_transform, FRAME_CONVENTION_VERSION),
        },
        "per_run": per_run,
        "pairwise_differences": pairwise,
        "cross_run": {
            "max_translation_difference_m": max_translation,
            "max_rotation_difference_deg": max_rotation,
            "translation_limit_strictly_lt_m": args.max_cross_run_translation_m,
            "rotation_limit_strictly_lt_deg": args.max_cross_run_rotation_deg,
            "consistent": cross_run_consistent,
        },
        "parameters": {
            "relative_interval_sec": args.relative_interval_sec,
            "relative_stride": args.relative_stride,
            "min_rotation_deg": args.min_rotation_deg,
            "max_interpolation_bracket_sec": args.max_interpolation_bracket_sec,
        },
    }
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"status={status} runs={len(per_run)} motions={shared['motion_count']} "
        f"max_translation_difference_m={max_translation:.6f} "
        f"max_rotation_difference_deg={max_rotation:.6f}"
    )
    print(f"output_json={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
