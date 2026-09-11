#!/usr/bin/env python3
"""Audit an M3DGR ground-truth text file against its ROS2 VRPN topic.

Expected text columns are ``timestamp x y z qx qy qz qw``.  This script is
deliberately an integrity/provenance check; passing it does not establish the
sensor-to-rigid-body extrinsic required for trajectory evaluation.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rosbag2_py
from geometry_msgs.msg import PoseStamped
from rclpy.serialization import deserialize_message


DEFAULT_GT_TOPIC = "/vrpn_client_node/UGV/pose"


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def load_gt(path):
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 8:
        raise ValueError(f"{path} must contain timestamp x y z qx qy qz qw")
    data = data[:, :8]
    if not np.all(np.isfinite(data)):
        indices = np.argwhere(~np.isfinite(data))
        raise ValueError(f"{path} contains non-finite values at {indices[:10].tolist()}")
    if data.shape[0] < 2:
        raise ValueError(f"{path} must contain at least two poses")
    return data


def safe_distribution(values):
    if values.size == 0:
        return {"min": None, "median": None, "p99": None, "max": None}
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(np.max(values)),
    }


def normalized_quaternions(quaternions):
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms <= 1.0e-12):
        bad = np.flatnonzero(norms <= 1.0e-12)
        raise ValueError(f"zero-norm quaternion at rows {bad[:10].tolist()}")
    return quaternions / norms[:, None], norms


def summarize_pose_sequence(data):
    times = data[:, 0]
    positions = data[:, 1:4]
    quaternions, quaternion_norms = normalized_quaternions(data[:, 4:8])
    dt = np.diff(times)
    positive_dt = dt[dt > 0.0]
    position_steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    valid_motion = dt > 0.0
    speeds = position_steps[valid_motion] / dt[valid_motion]
    adjacent_dots = np.sum(quaternions[1:] * quaternions[:-1], axis=1)
    angular_steps = 2.0 * np.arccos(np.clip(np.abs(adjacent_dots), 0.0, 1.0))
    angular_speeds = angular_steps[valid_motion] / dt[valid_motion]
    unique_count = int(np.unique(times).size)
    identity_distance = np.minimum(
        np.linalg.norm(quaternions - np.array([0.0, 0.0, 0.0, 1.0]), axis=1),
        np.linalg.norm(quaternions + np.array([0.0, 0.0, 0.0, 1.0]), axis=1),
    )
    return {
        "rows": int(data.shape[0]),
        "unique_timestamps": unique_count,
        "duplicate_timestamps": int(data.shape[0] - unique_count),
        "timestamp_inversions": int(np.count_nonzero(dt < 0.0)),
        "nonpositive_intervals": int(np.count_nonzero(dt <= 0.0)),
        "first_timestamp_s": float(times[0]),
        "last_timestamp_s": float(times[-1]),
        "duration_s": float(times[-1] - times[0]),
        "positive_interval_s": safe_distribution(positive_dt),
        "quaternion_norm": {
            "min": float(np.min(quaternion_norms)),
            "mean": float(np.mean(quaternion_norms)),
            "max": float(np.max(quaternion_norms)),
            "max_abs_error_from_one": float(np.max(np.abs(quaternion_norms - 1.0))),
        },
        "quaternion_raw_sign_flips": int(np.count_nonzero(adjacent_dots < 0.0)),
        "identity_quaternion_rows": int(np.count_nonzero(identity_distance <= 1.0e-9)),
        "position_min_m": np.min(positions, axis=0).tolist(),
        "position_max_m": np.max(positions, axis=0).tolist(),
        "path_length_m": float(np.sum(position_steps)),
        "position_step_m": safe_distribution(position_steps),
        "linear_speed_mps": safe_distribution(speeds),
        "angular_step_rad": safe_distribution(angular_steps),
        "angular_speed_radps": safe_distribution(angular_speeds),
    }


def read_bag_pose_topic(bag_path, topic):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if topic not in topic_types:
        raise ValueError(f"bag does not contain {topic}")
    expected_type = "geometry_msgs/msg/PoseStamped"
    if topic_types[topic] != expected_type:
        raise ValueError(f"{topic} has type {topic_types[topic]}, expected {expected_type}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    rows = []
    storage_times = []
    frame_ids = set()
    while reader.has_next():
        _, serialized, storage_ns = reader.read_next()
        message = deserialize_message(serialized, PoseStamped)
        pose = message.pose
        rows.append(
            [
                stamp_seconds(message.header.stamp),
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
        )
        storage_times.append(storage_ns * 1.0e-9)
        frame_ids.add(message.header.frame_id)
    if not rows:
        raise ValueError(f"{topic} contains no messages")
    return np.asarray(rows, dtype=float), np.asarray(storage_times), sorted(frame_ids)


def nearest_indices(reference_times, query_times):
    right = np.searchsorted(reference_times, query_times, side="left")
    right = np.clip(right, 0, len(reference_times) - 1)
    left = np.clip(right - 1, 0, len(reference_times) - 1)
    choose_left = np.abs(query_times - reference_times[left]) <= np.abs(
        query_times - reference_times[right]
    )
    return np.where(choose_left, left, right)


def pose_errors(first, second):
    position_errors = np.linalg.norm(first[:, 1:4] - second[:, 1:4], axis=1)
    first_q, _ = normalized_quaternions(first[:, 4:8])
    second_q, _ = normalized_quaternions(second[:, 4:8])
    dots = np.sum(first_q * second_q, axis=1)
    angular_errors = 2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0))
    return position_errors, angular_errors


def compare_sequences(text_data, bag_data, tolerance_s):
    direct = None
    if text_data.shape[0] == bag_data.shape[0]:
        position_errors, angular_errors = pose_errors(text_data, bag_data)
        direct = {
            "row_count_equal": True,
            "max_abs_timestamp_delta_s": float(np.max(np.abs(text_data[:, 0] - bag_data[:, 0]))),
            "position_error_m": safe_distribution(position_errors),
            "orientation_error_rad": safe_distribution(angular_errors),
        }
    else:
        direct = {"row_count_equal": False}

    order = np.argsort(bag_data[:, 0], kind="stable")
    sorted_bag = bag_data[order]
    indices = nearest_indices(sorted_bag[:, 0], text_data[:, 0])
    timestamp_errors = np.abs(text_data[:, 0] - sorted_bag[indices, 0])
    matched = timestamp_errors <= tolerance_s
    matched_count = int(np.count_nonzero(matched))
    nearest = {
        "tolerance_s": tolerance_s,
        "text_rows_matched": matched_count,
        "text_rows_matched_fraction": float(matched_count / text_data.shape[0]),
        "max_abs_timestamp_delta_for_matches_s": (
            float(np.max(timestamp_errors[matched])) if matched_count else None
        ),
        "position_error_m": None,
        "orientation_error_rad": None,
    }
    if matched_count:
        position_errors, angular_errors = pose_errors(text_data[matched], sorted_bag[indices[matched]])
        nearest["position_error_m"] = safe_distribution(position_errors)
        nearest["orientation_error_rad"] = safe_distribution(angular_errors)
    return {"direct_row_comparison": direct, "nearest_timestamp_comparison": nearest}


def audit(gt_path, bag_path, topic, timestamp_tolerance_s, position_tolerance_m, orientation_tolerance_rad):
    text_data = load_gt(gt_path)
    bag_data, storage_times, frame_ids = read_bag_pose_topic(bag_path, topic)
    report = {
        "schema_version": 1,
        "scope": "GT text integrity and equality/proximity to bag PoseStamped; not sensor extrinsic validation",
        "gt_path": str(gt_path.resolve()),
        "bag_path": str(bag_path.resolve()),
        "bag_topic": topic,
        "text": summarize_pose_sequence(text_data),
        "bag_topic_data": summarize_pose_sequence(bag_data),
        "bag_storage": {
            "first_timestamp_s": float(storage_times[0]),
            "last_timestamp_s": float(storage_times[-1]),
            "duration_s": float(storage_times[-1] - storage_times[0]),
            "frame_ids": frame_ids,
        },
        "text_to_bag": compare_sequences(text_data, bag_data, timestamp_tolerance_s),
    }
    direct = report["text_to_bag"]["direct_row_comparison"]
    rejection_reasons = []
    if not direct["row_count_equal"]:
        rejection_reasons.append("row_count_mismatch")
    else:
        if direct["max_abs_timestamp_delta_s"] > timestamp_tolerance_s:
            rejection_reasons.append("timestamp_difference_exceeds_tolerance")
        if direct["position_error_m"]["max"] > position_tolerance_m:
            rejection_reasons.append("position_difference_exceeds_tolerance")
        if direct["orientation_error_rad"]["max"] > orientation_tolerance_rad:
            rejection_reasons.append("orientation_difference_exceeds_tolerance")
    report["acceptance"] = {
        "text_matches_bag_topic": not rejection_reasons,
        "timestamp_tolerance_s": timestamp_tolerance_s,
        "position_tolerance_m": position_tolerance_m,
        "orientation_tolerance_rad": orientation_tolerance_rad,
        "rejection_reasons": rejection_reasons,
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gt", type=Path)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--topic", default=DEFAULT_GT_TOPIC)
    parser.add_argument("--timestamp-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--position-tolerance", type=float, default=5.0e-6)
    parser.add_argument("--orientation-tolerance", type=float, default=5.0e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.gt.is_file():
        raise FileNotFoundError(args.gt)
    if not args.bag.is_dir():
        raise FileNotFoundError(args.bag)
    tolerances = {
        "timestamp tolerance": args.timestamp_tolerance,
        "position tolerance": args.position_tolerance,
        "orientation tolerance": args.orientation_tolerance,
    }
    for name, value in tolerances.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    report = audit(
        args.gt,
        args.bag,
        args.topic,
        args.timestamp_tolerance,
        args.position_tolerance,
        args.orientation_tolerance,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["acceptance"]["text_matches_bag_topic"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
