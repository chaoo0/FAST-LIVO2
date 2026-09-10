#!/usr/bin/env python3
"""Position-only M3DGR evaluation with explicit interpolation and SE(3) alignment."""

import argparse
import json
from pathlib import Path

import numpy as np


def load_tum_positions(path):
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 4:
        raise ValueError(f"{path} must contain timestamp x y z columns")
    if not np.all(np.isfinite(data[:, :4])):
        raise ValueError(f"{path} contains non-finite timestamp or position")
    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]
    times, first_indices = np.unique(data[:, 0], return_index=True)
    positions = np.vstack(
        [data[data[:, 0] == time, 1:4].mean(axis=0) for time in times]
    )
    return times, positions, data.shape[0] - len(first_indices)


def interpolate_positions(source_times, source_positions, query_times, max_bracket_gap):
    right = np.searchsorted(source_times, query_times, side="left")
    valid = (right > 0) & (right < len(source_times))
    left = np.clip(right - 1, 0, len(source_times) - 1)
    right = np.clip(right, 0, len(source_times) - 1)
    bracket = source_times[right] - source_times[left]
    valid &= bracket > 0.0
    valid &= bracket <= max_bracket_gap
    alpha = np.zeros_like(query_times, dtype=float)
    alpha[valid] = (query_times[valid] - source_times[left[valid]]) / bracket[valid]
    valid &= (alpha >= 0.0) & (alpha <= 1.0)
    interpolated = source_positions[left] + alpha[:, None] * (source_positions[right] - source_positions[left])
    return interpolated, valid


def rigid_alignment(source, target):
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def position_rpe(times, estimate, reference, horizon, tolerance):
    errors = []
    for index, start_time in enumerate(times):
        target = start_time + horizon
        end_index = int(np.searchsorted(times, target))
        candidates = [candidate for candidate in (end_index - 1, end_index) if index < candidate < len(times)]
        if not candidates:
            continue
        best = min(candidates, key=lambda candidate: abs(times[candidate] - target))
        if abs(times[best] - target) > tolerance:
            continue
        estimate_delta = estimate[best] - estimate[index]
        reference_delta = reference[best] - reference[index]
        errors.append(float(np.linalg.norm(estimate_delta - reference_delta)))
    if not errors:
        return {"pairs": 0, "rmse_m": None, "mean_m": None, "median_m": None, "max_m": None}
    values = np.asarray(errors)
    return {
        "pairs": len(errors),
        "rmse_m": float(np.sqrt(np.mean(values**2))),
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "max_m": float(np.max(values)),
    }


def evaluate(gt_path, estimate_path, max_bracket_gap, horizons, horizon_tolerance):
    gt_times, gt_positions, gt_duplicates = load_tum_positions(gt_path)
    estimate_times, estimate_positions, estimate_duplicates = load_tum_positions(estimate_path)
    interpolated_gt, valid = interpolate_positions(gt_times, gt_positions, estimate_times, max_bracket_gap)
    if np.count_nonzero(valid) < 3:
        raise ValueError("fewer than three estimate samples have valid GT brackets")
    matched_times = estimate_times[valid]
    matched_estimate = estimate_positions[valid]
    matched_gt = interpolated_gt[valid]
    rotation, translation = rigid_alignment(matched_estimate, matched_gt)
    aligned_estimate = (rotation @ matched_estimate.T).T + translation
    ate_values = np.linalg.norm(aligned_estimate - matched_gt, axis=1)
    quaternion_warning = None
    gt_raw = np.loadtxt(gt_path, comments="#", ndmin=2)
    if gt_raw.shape[1] >= 8 and np.allclose(gt_raw[:, 4:8], np.array([0.0, 0.0, 0.0, 1.0])):
        quaternion_warning = "GT orientations are all identity; rotation and full SE(3) RPE are not evaluated."
    return {
        "schema_version": 1,
        "metric_scope": "position_only",
        "alignment": "rigid SE(3) fit on matched positions, scale fixed to 1",
        "interpolation": "linear GT position interpolation at estimate timestamps",
        "max_gt_bracket_gap_s": max_bracket_gap,
        "horizon_match_tolerance_s": horizon_tolerance,
        "gt_samples_raw": int(len(gt_raw)),
        "gt_duplicate_timestamps_removed": int(gt_duplicates),
        "estimate_samples": int(len(estimate_times)),
        "estimate_duplicate_timestamps_removed": int(estimate_duplicates),
        "matched_samples": int(len(matched_times)),
        "matched_fraction": float(len(matched_times) / len(estimate_times)),
        "ate": {
            "rmse_m": float(np.sqrt(np.mean(ate_values**2))),
            "mean_m": float(np.mean(ate_values)),
            "median_m": float(np.median(ate_values)),
            "max_m": float(np.max(ate_values)),
        },
        "position_rpe": {
            f"{horizon:g}s": position_rpe(
                matched_times, aligned_estimate, matched_gt, horizon, horizon_tolerance
            )
            for horizon in horizons
        },
        "alignment_rotation_row_major": rotation.reshape(-1).tolist(),
        "alignment_translation_m": translation.tolist(),
        "warning": quaternion_warning,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gt", type=Path)
    parser.add_argument("estimate", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-gt-bracket-gap", type=float)
    parser.add_argument("--horizon", type=float, action="append", default=[])
    parser.add_argument("--horizon-tolerance", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = {}
    if args.config:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if config.get("metric_scope") != "position_only":
            raise ValueError("only a position_only evaluation configuration is supported")
    max_bracket_gap = (
        args.max_gt_bracket_gap
        if args.max_gt_bracket_gap is not None
        else float(config.get("max_gt_bracket_gap_s", 0.25))
    )
    horizon_tolerance = (
        args.horizon_tolerance
        if args.horizon_tolerance is not None
        else float(config.get("horizon_match_tolerance_s", 0.06))
    )
    horizons = args.horizon or config.get("rpe_horizons_s", [1.0, 5.0, 10.0])
    report = evaluate(args.gt, args.estimate, max_bracket_gap, horizons, horizon_tolerance)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
