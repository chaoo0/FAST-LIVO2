#!/usr/bin/env python3
"""Rewrite an old Livox CustomMsg ROS2 bag into a driver2 LIO-only bag.

The old and driver2 Livox CustomMsg definitions have the same CDR field
layout.  This tool copies only MID360 lidar and IMU messages, changes the
LiDAR connection type, preserves serialized point data/timestamps verbatim,
and validates sampled ``offset_time`` values before and after the rewrite.
It intentionally does not modify FAST-LIVO2 point preprocessing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader, Writer
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from prepare_m3dgr_bag import (
    CUSTOM_MSG_TEMPLATE,
    CUSTOM_POINT_MSG,
    DRIVER2_LIVOX_TYPE,
    MID360_IMU_TOPIC,
    MID360_LIDAR_TOPIC,
    OLD_LIVOX_TYPE,
    inspect_ros2_bag,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bag", type=Path, required=True)
    parser.add_argument("--output-bag", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--lidar-topic", default=MID360_LIDAR_TOPIC)
    parser.add_argument("--imu-topic", default=MID360_IMU_TOPIC)
    parser.add_argument("--sample-lidar-messages", type=int, default=20)
    return parser.parse_args()


def build_store():
    store = get_typestore(Stores.ROS2_HUMBLE)
    for package in ("livox_ros_driver", "livox_ros_driver2"):
        store.register(get_types_from_msg(CUSTOM_POINT_MSG, f"{package}/msg/CustomPoint"))
        store.register(
            get_types_from_msg(
                CUSTOM_MSG_TEMPLATE.format(package=package),
                f"{package}/msg/CustomMsg",
            )
        )
    return store


def point_signature(message) -> dict[str, int | bool]:
    offsets = np.asarray([point.offset_time for point in message.points], dtype=np.int64)
    return {
        "point_num_field": int(message.point_num),
        "point_array_length": len(offsets),
        "first_offset_time": int(offsets[0]) if len(offsets) else 0,
        "last_offset_time": int(offsets[-1]) if len(offsets) else 0,
        "max_offset_time": int(offsets.max()) if len(offsets) else 0,
        "offsets_monotonic": bool(len(offsets) < 2 or np.all(np.diff(offsets) >= 0)),
    }


def main() -> int:
    args = parse_args()
    source = args.input_bag.expanduser().resolve()
    destination = args.output_bag.expanduser().resolve()
    report_path = args.report_json.expanduser().resolve()
    if not source.is_dir() or not (source / "metadata.yaml").is_file():
        raise SystemExit(f"Input is not a ROS2 bag directory: {source}")
    if destination.exists():
        raise SystemExit(f"Destination already exists; refusing to overwrite: {destination}")
    if args.sample_lidar_messages <= 0:
        raise SystemExit("--sample-lidar-messages must be positive.")

    store = build_store()
    source_signatures: list[dict[str, int | bool]] = []
    counts = {args.lidar_topic: 0, args.imu_topic: 0}
    with Reader(source) as reader:
        connection_by_topic = {connection.topic: connection for connection in reader.connections}
        missing = [
            topic for topic in (args.lidar_topic, args.imu_topic) if topic not in connection_by_topic
        ]
        if missing:
            raise SystemExit(f"Input bag is missing required topics: {', '.join(missing)}")
        lidar_connection = connection_by_topic[args.lidar_topic]
        imu_connection = connection_by_topic[args.imu_topic]
        if lidar_connection.msgtype == DRIVER2_LIVOX_TYPE:
            raise SystemExit("Input already uses livox_ros_driver2/msg/CustomMsg; rewrite is unnecessary.")
        if lidar_connection.msgtype != OLD_LIVOX_TYPE:
            raise SystemExit(
                f"Expected {OLD_LIVOX_TYPE}, found {lidar_connection.msgtype}."
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        with Writer(destination, version=8) as writer:
            output_lidar = writer.add_connection(
                args.lidar_topic,
                DRIVER2_LIVOX_TYPE,
                typestore=store,
                serialization_format=lidar_connection.ext.serialization_format,
                offered_qos_profiles=lidar_connection.ext.offered_qos_profiles,
            )
            output_imu = writer.add_connection(
                args.imu_topic,
                imu_connection.msgtype,
                typestore=store,
                serialization_format=imu_connection.ext.serialization_format,
                offered_qos_profiles=imu_connection.ext.offered_qos_profiles,
            )
            selected = [lidar_connection, imu_connection]
            for connection, timestamp, raw in reader.messages(connections=selected):
                if connection.topic == args.lidar_topic:
                    if len(source_signatures) < args.sample_lidar_messages:
                        old_message = store.deserialize_cdr(raw, OLD_LIVOX_TYPE)
                        driver2_message = store.deserialize_cdr(raw, DRIVER2_LIVOX_TYPE)
                        old_signature = point_signature(old_message)
                        driver2_signature = point_signature(driver2_message)
                        if old_signature != driver2_signature:
                            raise SystemExit(
                                "Old/driver2 CDR layout validation failed; no output is trusted."
                            )
                        source_signatures.append(old_signature)
                    writer.write(output_lidar, timestamp, raw)
                else:
                    writer.write(output_imu, timestamp, raw)
                counts[connection.topic] += 1

    output_audit = inspect_ros2_bag(
        destination, args.lidar_topic, args.imu_topic, args.sample_lidar_messages
    )
    passed = bool(
        counts[args.lidar_topic] > 0
        and counts[args.imu_topic] > 0
        and output_audit["driver_compatibility"] == "direct_fast_livo2"
        and output_audit["point_time_audit"]["per_point_time_offsets_preserved"]
    )
    payload = {
        "schema_version": 1,
        "input_bag": str(source),
        "output_bag": str(destination),
        "scope": "LIO-only: MID360 lidar plus built-in IMU",
        "source_lidar_type": OLD_LIVOX_TYPE,
        "output_lidar_type": DRIVER2_LIVOX_TYPE,
        "serialized_payload_copy": "verbatim CDR bytes",
        "message_counts": counts,
        "sampled_source_point_signatures": source_signatures,
        "output_audit": output_audit,
        "passed": passed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(f"Driver2 rewrite audit failed; inspect {report_path}")
    print(
        f"passed=true lidar_messages={counts[args.lidar_topic]} "
        f"imu_messages={counts[args.imu_topic]} output_bag={destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
