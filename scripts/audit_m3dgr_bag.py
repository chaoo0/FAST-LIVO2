#!/usr/bin/env python3
"""Audit ROS2 M3DGR bags with SequentialReader and deserialize selected topics."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


DEFAULT_TOPICS = {
    "/livox/mid360/lidar",
    "/livox/mid360/imu",
    "/camera/color/image_raw/compressed",
    "/odom",
}


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def audit_bag(bag_path, selected_topics, full_lidar_deserialize=False):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    topics = {item.name: item.type for item in reader.get_all_topics_and_types()}
    installed_types = {}
    unavailable_types = {}
    for topic, type_name in topics.items():
        try:
            installed_types[topic] = get_message(type_name)
        except (AttributeError, ImportError, ModuleNotFoundError, ValueError) as error:
            unavailable_types[topic] = f"{type(error).__name__}: {error}"

    target_topics = set(topics) if selected_topics == {"*"} else set(selected_topics)
    missing_requested_topics = sorted(target_topics - set(topics))
    target_topics &= set(topics)
    reader.set_filter(rosbag2_py.StorageFilter(topics=sorted(target_topics)))

    stats = defaultdict(
        lambda: {
            "count": 0,
            "deserialized": 0,
            "deserialize_failures": 0,
            "storage_first": None,
            "storage_last": None,
            "header_first": None,
            "header_last": None,
            "header_inversions": 0,
            "max_header_gap": 0.0,
            "frame_ids": set(),
            "point_num_min": None,
            "point_num_max": None,
            "line_max": None,
            "scan_offset_max_ms": None,
            "lidar_payloads_inspected": 0,
            "serialized_records_read": 0,
        }
    )

    while reader.has_next():
        topic, serialized, storage_ns = reader.read_next()
        item = stats[topic]
        item["serialized_records_read"] += 1
        storage_time = storage_ns * 1.0e-9
        item["count"] += 1
        item["storage_first"] = storage_time if item["storage_first"] is None else item["storage_first"]
        item["storage_last"] = storage_time
        message_class = installed_types.get(topic)
        if message_class is None:
            continue
        is_livox_cloud = topics[topic] == "livox_ros_driver2/msg/CustomMsg"
        if is_livox_cloud and not full_lidar_deserialize and item["deserialized"] >= 20:
            continue
        try:
            message = deserialize_message(serialized, message_class)
        except Exception:  # audit records failure count; it does not hide the failure
            item["deserialize_failures"] += 1
            continue
        item["deserialized"] += 1
        if hasattr(message, "header"):
            current = stamp_seconds(message.header.stamp)
            previous = item["header_last"]
            if previous is not None:
                gap = current - previous
                if gap < 0.0:
                    item["header_inversions"] += 1
                item["max_header_gap"] = max(item["max_header_gap"], gap)
            item["header_first"] = current if item["header_first"] is None else item["header_first"]
            item["header_last"] = current
            if message.header.frame_id:
                item["frame_ids"].add(message.header.frame_id)
        if hasattr(message, "point_num") and hasattr(message, "points"):
            point_num = int(message.point_num)
            item["point_num_min"] = point_num if item["point_num_min"] is None else min(item["point_num_min"], point_num)
            item["point_num_max"] = point_num if item["point_num_max"] is None else max(item["point_num_max"], point_num)
            # Full messages are deserialized and counted. Point-wise line/time
            # inspection is intentionally bounded because a single bag contains
            # tens of millions of Livox points.
            if message.points and item["lidar_payloads_inspected"] < 20:
                line_max = max(int(point.line) for point in message.points)
                offset_max_ms = max(int(point.offset_time) for point in message.points) * 1.0e-6
                item["line_max"] = line_max if item["line_max"] is None else max(item["line_max"], line_max)
                item["scan_offset_max_ms"] = (
                    offset_max_ms
                    if item["scan_offset_max_ms"] is None
                    else max(item["scan_offset_max_ms"], offset_max_ms)
                )
                item["lidar_payloads_inspected"] += 1

    fully_deserialized = True
    output_topics = {}
    for topic in sorted(target_topics):
        item = stats[topic]
        duration = None
        frequency = None
        storage_duration = None
        storage_frequency = None
        if item["header_first"] is not None and item["header_last"] > item["header_first"]:
            duration = item["header_last"] - item["header_first"]
            frequency = (item["deserialized"] - 1) / duration
        if item["storage_first"] is not None and item["storage_last"] > item["storage_first"]:
            storage_duration = item["storage_last"] - item["storage_first"]
            storage_frequency = (item["count"] - 1) / storage_duration
        item["frame_ids"] = sorted(item["frame_ids"])
        item["header_duration"] = duration
        item["mean_deserialized_header_frequency_hz"] = frequency
        item["storage_duration"] = storage_duration
        item["mean_storage_frequency_hz"] = storage_frequency
        item["type"] = topics[topic]
        item["deserialization_scope"] = (
            "all"
            if item["deserialized"] == item["count"]
            else f"sample:{item['deserialized']}"
        )
        if topic not in installed_types or item["deserialized"] != item["count"] or item["deserialize_failures"]:
            fully_deserialized = False
        output_topics[topic] = item

    return {
        "schema_version": 1,
        "bag_path": str(Path(bag_path).resolve()),
        "requested_topics": sorted(target_topics),
        "missing_requested_topics": missing_requested_topics,
        "all_selected_serialized_records_read": all(
            stats[topic]["serialized_records_read"] == stats[topic]["count"] for topic in target_topics
        ),
        "selected_topics_fully_deserialized": fully_deserialized and not missing_requested_topics,
        "unavailable_topic_types": unavailable_types,
        "topics": output_topics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--topic", action="append", dest="topics")
    parser.add_argument("--all-topics", action="store_true")
    parser.add_argument(
        "--full-lidar-deserialize",
        action="store_true",
        help="deserialize every Livox point cloud; slow for multi-gigabyte bags",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    selected = {"*"} if args.all_topics else set(args.topics or DEFAULT_TOPICS)
    report = audit_bag(args.bag, selected, args.full_lidar_deserialize)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["all_selected_serialized_records_read"] and not report["missing_requested_topics"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
