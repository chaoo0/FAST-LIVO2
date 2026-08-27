#!/usr/bin/env python3
"""Compare current vs Kalibr-variance LIO profiles on frozen train bags only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scipy.spatial.transform import Rotation

from mamba_pose_se3 import invert_transform, transforms_from_poses


TRAIN_RUNS = (
    "Dynamic01",
    "Occlusion01",
    "Varying-illu01",
    "Dark03",
    "Wheel-float01",
    "Sha-turn01",
)
PROFILES = ("current", "kalibr_variance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        action="append",
        required=True,
        metavar="PROFILE:RUN_ID:ALIGNED_CSV",
        help="Repeat for both profiles and every frozen training run.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--freeze-output-yaml", type=Path)
    parser.add_argument("--rpe-delta-sec", type=float, default=1.0)
    parser.add_argument("--rpe-time-tolerance-sec", type=float, default=0.1)
    return parser.parse_args()


def parse_trajectory_specs(specs: list[str]) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {profile: {} for profile in PROFILES}
    for spec in specs:
        fields = spec.split(":", 2)
        if len(fields) != 3:
            raise SystemExit(f"Invalid --trajectory '{spec}'.")
        profile, run_id, raw_path = fields
        if profile not in PROFILES:
            raise SystemExit(f"Unknown profile '{profile}'. Expected one of {PROFILES}.")
        if run_id not in TRAIN_RUNS:
            raise SystemExit(f"{run_id} is not in the frozen training split.")
        if run_id in result[profile]:
            raise SystemExit(f"Duplicate trajectory for {profile}:{run_id}.")
        result[profile][run_id] = Path(raw_path).expanduser().resolve()
    for profile in PROFILES:
        missing = sorted(set(TRAIN_RUNS) - set(result[profile]))
        if missing:
            raise SystemExit(f"Profile {profile} is missing training runs: {missing}")
    return result


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def load_errors(path: Path, run_id: str, rpe_delta: float, tolerance: float) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"Aligned CSV not found: {path}")
    frame = pd.read_csv(path)
    required = [
        "timestamp",
        "pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "rot_w",
        "gt_pos_x", "gt_pos_y", "gt_pos_z", "gt_rot_x", "gt_rot_y", "gt_rot_z", "gt_rot_w",
        "gt_valid", "gt_status", "run_id", "segment_id",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {missing}")
    valid = pd.to_numeric(frame["gt_valid"], errors="coerce").fillna(0).to_numpy() == 1
    valid &= frame["gt_status"].astype(str).to_numpy() == "trusted_gt"
    valid &= frame["run_id"].astype(str).to_numpy() == run_id
    selected = frame.loc[valid].copy()
    if len(selected) < 3:
        raise SystemExit(f"{run_id} has too few trusted GT rows in {path}.")
    numeric_columns = [column for column in required if column not in ("gt_status", "run_id")]
    numeric = selected[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise SystemExit(f"{run_id} contains non-finite trusted pose fields.")
    timestamps = numeric["timestamp"].to_numpy(dtype=np.float64)
    segments = numeric["segment_id"].to_numpy(dtype=np.int64)
    est_pose = numeric[["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "rot_w"]].to_numpy()
    ref_pose = numeric[["gt_pos_x", "gt_pos_y", "gt_pos_z", "gt_rot_x", "gt_rot_y", "gt_rot_z", "gt_rot_w"]].to_numpy()
    est = transforms_from_poses(est_pose[:, :3], est_pose[:, 3:])
    ref = transforms_from_poses(ref_pose[:, :3], ref_pose[:, 3:])
    absolute_error = np.stack([invert_transform(e) @ r for e, r in zip(est, ref)])
    absolute_translation = np.linalg.norm(absolute_error[:, :3, 3], axis=1)
    absolute_rotation = np.linalg.norm(
        Rotation.from_matrix(absolute_error[:, :3, :3]).as_rotvec(), axis=1
    )

    relative_translation_errors: list[float] = []
    relative_rotation_errors: list[float] = []
    for first in range(len(timestamps)):
        target = timestamps[first] + rpe_delta
        second = int(np.searchsorted(timestamps, target))
        candidates = [index for index in (second - 1, second) if first < index < len(timestamps)]
        if not candidates:
            continue
        second = min(candidates, key=lambda index: abs(timestamps[index] - target))
        if abs(timestamps[second] - target) > tolerance or segments[first] != segments[second]:
            continue
        est_relative = invert_transform(est[first]) @ est[second]
        ref_relative = invert_transform(ref[first]) @ ref[second]
        relative_error = invert_transform(est_relative) @ ref_relative
        relative_translation_errors.append(float(np.linalg.norm(relative_error[:3, 3])))
        relative_rotation_errors.append(
            float(
                np.linalg.norm(
                    Rotation.from_matrix(relative_error[:3, :3]).as_rotvec()
                )
            )
        )
    if not relative_translation_errors:
        raise SystemExit(f"{run_id} produced no RPE pairs at delta={rpe_delta}s.")
    return {
        "ate_translation_norm": absolute_translation,
        "ate_rotation_norm_rad": absolute_rotation,
        "rpe_translation_norm": np.asarray(relative_translation_errors),
        "rpe_rotation_norm_rad": np.asarray(relative_rotation_errors),
        "frame_count": len(absolute_error),
        "rpe_pair_count": len(relative_translation_errors),
    }


def summarize(errors: dict[str, object]) -> dict[str, float | int]:
    return {
        "frame_count": int(errors["frame_count"]),
        "rpe_pair_count": int(errors["rpe_pair_count"]),
        "ate_translation_rmse_m": rmse(errors["ate_translation_norm"]),
        "ate_rotation_rmse_deg": rmse(np.degrees(errors["ate_rotation_norm_rad"])),
        "rpe_translation_rmse_m": rmse(errors["rpe_translation_norm"]),
        "rpe_rotation_rmse_deg": rmse(np.degrees(errors["rpe_rotation_norm_rad"])),
    }


def main() -> int:
    args = parse_args()
    if args.rpe_delta_sec <= 0 or args.rpe_time_tolerance_sec <= 0:
        raise SystemExit("RPE delta and tolerance must be positive.")
    trajectories = parse_trajectory_specs(args.trajectory)
    payload: dict[str, object] = {
        "schema_version": 1,
        "selection_scope": "frozen_training_sequences_only",
        "training_runs": list(TRAIN_RUNS),
        "profiles": {},
        "rpe_delta_sec": args.rpe_delta_sec,
        "rpe_time_tolerance_sec": args.rpe_time_tolerance_sec,
    }
    aggregate_metrics: dict[str, dict[str, float | int]] = {}
    for profile in PROFILES:
        per_run: dict[str, object] = {}
        aggregate_parts: dict[str, list[np.ndarray]] = {
            "ate_translation_norm": [],
            "ate_rotation_norm_rad": [],
            "rpe_translation_norm": [],
            "rpe_rotation_norm_rad": [],
        }
        total_frames = 0
        total_pairs = 0
        for run_id in TRAIN_RUNS:
            errors = load_errors(
                trajectories[profile][run_id], run_id, args.rpe_delta_sec, args.rpe_time_tolerance_sec
            )
            per_run[run_id] = summarize(errors)
            total_frames += int(errors["frame_count"])
            total_pairs += int(errors["rpe_pair_count"])
            for key in aggregate_parts:
                aggregate_parts[key].append(errors[key])
        aggregate = {key: np.concatenate(parts) for key, parts in aggregate_parts.items()}
        aggregate["frame_count"] = total_frames
        aggregate["rpe_pair_count"] = total_pairs
        aggregate_metrics[profile] = summarize(aggregate)
        payload["profiles"][profile] = {
            "per_run": per_run,
            "aggregate": aggregate_metrics[profile],
        }
    selected = min(
        PROFILES,
        key=lambda profile: (
            aggregate_metrics[profile]["ate_translation_rmse_m"],
            aggregate_metrics[profile]["rpe_translation_rmse_m"],
            aggregate_metrics[profile]["ate_rotation_rmse_deg"],
            aggregate_metrics[profile]["rpe_rotation_rmse_deg"],
        ),
    )
    payload["selection_rule"] = (
        "lexicographic minimum of aggregate translation ATE, translation RPE, "
        "rotation ATE, rotation RPE; all metrics are RMSE"
    )
    payload["selected_profile"] = selected
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output.write_text(encoded, encoding="utf-8")
    if args.freeze_output_yaml:
        freeze_path = args.freeze_output_yaml.expanduser().resolve()
        freeze_path.parent.mkdir(parents=True, exist_ok=True)
        freeze_payload = {
            "selection_status": "frozen_from_training_only",
            "selected_profile": selected,
            "training_runs": list(TRAIN_RUNS),
            "comparison_report": str(output),
            "comparison_report_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "validation_or_test_used_for_selection": False,
        }
        freeze_path.write_text(yaml.safe_dump(freeze_payload, sort_keys=False), encoding="utf-8")
    print(f"selected_profile={selected} report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
