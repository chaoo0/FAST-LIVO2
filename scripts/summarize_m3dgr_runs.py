#!/usr/bin/env python3
"""Summarize repeated FAST-LIVO2 M3DGR runs without selecting a best run."""

import argparse
import json
import re
from pathlib import Path

import numpy as np


CURRENT_TOTAL = re.compile(r"Current Total Time\s+\|\s+([0-9.eE+-]+)")
AVERAGE_TOTAL = re.compile(r"Average Total Time\s+\|\s+([0-9.eE+-]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def load_key_values(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def rotation_distances_xyzw(first, second):
    first = first / np.linalg.norm(first, axis=1)[:, None]
    second = second / np.linalg.norm(second, axis=1)[:, None]
    dots = np.abs(np.sum(first * second, axis=1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def run_report(directory):
    trajectory = np.loadtxt(directory / "trajectory.tum", ndmin=2)
    metrics = json.loads((directory / "position_metrics.json").read_text(encoding="utf-8"))
    summary = load_key_values(directory / "run_summary.txt")
    raw_log = (directory / "fast_livo.log").read_text(encoding="utf-8", errors="replace")
    log = ANSI.sub("", raw_log)
    current_times = np.asarray([float(value) for value in CURRENT_TOTAL.findall(log)])
    average_times = [float(value) for value in AVERAGE_TOTAL.findall(log)]
    timing = {
        "samples": int(len(current_times)),
        "mean_ms": float(np.mean(current_times) * 1000.0),
        "median_ms": float(np.median(current_times) * 1000.0),
        "p95_ms": float(np.quantile(current_times, 0.95) * 1000.0),
        "p99_ms": float(np.quantile(current_times, 0.99) * 1000.0),
        "max_ms": float(np.max(current_times) * 1000.0),
        "reported_final_average_ms": float(average_times[-1] * 1000.0),
    }
    return trajectory, {
        "directory": str(directory.resolve()),
        "trajectory_count": int(len(trajectory)),
        "first_timestamp": float(trajectory[0, 0]),
        "last_timestamp": float(trajectory[-1, 0]),
        "wall_duration_s": int(summary["wall_duration_s"]),
        "lio_map_updates": log.count("[ LIO ] Update Voxel Map"),
        "initial_empty_point_updates": log.count("[ LIO ]: No point!!!"),
        "shutdown_sigsegv": "exit code -11" in log,
        "imu_loopbacks": log.count("imu loop back"),
        "lidar_loopbacks": log.count("lidar loop back"),
        "out_sync_messages": log.count("out sync"),
        "timing": timing,
        "position_metrics": metrics,
    }


def coefficient_of_variation(values):
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1) / np.mean(values)) if len(values) > 1 else None


def summarize(run_directories):
    trajectories = []
    runs = []
    for directory in run_directories:
        trajectory, report = run_report(directory)
        trajectories.append(trajectory)
        runs.append(report)
    comparisons = []
    for first_index in range(len(trajectories)):
        for second_index in range(first_index + 1, len(trajectories)):
            first = trajectories[first_index]
            second = trajectories[second_index]
            if first.shape != second.shape:
                comparisons.append(
                    {
                        "runs": [first_index + 1, second_index + 1],
                        "same_shape": False,
                    }
                )
                continue
            position = np.linalg.norm(first[:, 1:4] - second[:, 1:4], axis=1)
            rotation = rotation_distances_xyzw(first[:, 4:8], second[:, 4:8])
            comparisons.append(
                {
                    "runs": [first_index + 1, second_index + 1],
                    "same_shape": True,
                    "max_timestamp_difference_s": float(np.max(np.abs(first[:, 0] - second[:, 0]))),
                    "position_rmse_m": float(np.sqrt(np.mean(position**2))),
                    "position_max_m": float(np.max(position)),
                    "rotation_rmse_rad": float(np.sqrt(np.mean(rotation**2))),
                    "rotation_max_rad": float(np.max(rotation)),
                }
            )
    ate = [run["position_metrics"]["ate"]["rmse_m"] for run in runs]
    rpe = {
        horizon: [run["position_metrics"]["position_rpe"][horizon]["rmse_m"] for run in runs]
        for horizon in ("1s", "5s", "10s")
    }
    return {
        "schema_version": 1,
        "runs": runs,
        "pairwise_repeatability": comparisons,
        "metric_repeatability": {
            "position_ate_rmse_m": ate,
            "position_ate_sample_cv": coefficient_of_variation(ate),
            "position_rpe_rmse_m": rpe,
            "position_rpe_sample_cv": {
                horizon: coefficient_of_variation(values) for horizon, values in rpe.items()
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.run_directory)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
