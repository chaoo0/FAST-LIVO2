#!/usr/bin/env python3
"""Verify, convert, and audit M3DGR bags for the ROS2 Humble LIO pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml
from rosbags.rosbag1 import Reader as Rosbag1Reader
from rosbags.rosbag2 import Reader as Rosbag2Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore


MID360_LIDAR_TOPIC = "/livox/mid360/lidar"
MID360_IMU_TOPIC = "/livox/mid360/imu"
OLD_LIVOX_TYPE = "livox_ros_driver/msg/CustomMsg"
DRIVER2_LIVOX_TYPE = "livox_ros_driver2/msg/CustomMsg"

CUSTOM_POINT_MSG = """uint32 offset_time
float32 x
float32 y
float32 z
uint8 reflectivity
uint8 tag
uint8 line
"""
CUSTOM_MSG_TEMPLATE = """std_msgs/Header header
uint64 timebase
uint32 point_num
uint8 lidar_id
uint8[3] rsvd
{package}/CustomPoint[] points
"""


def build_typestore():
    store = get_typestore(Stores.ROS2_HUMBLE)
    for package in ("livox_ros_driver", "livox_ros_driver2"):
        store.register(
            get_types_from_msg(CUSTOM_POINT_MSG, f"{package}/msg/CustomPoint")
        )
        store.register(
            get_types_from_msg(
                CUSTOM_MSG_TEMPLATE.format(package=package),
                f"{package}/msg/CustomMsg",
            )
        )
    return store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="Verify a ROS1 bag against the frozen manifest SHA256.")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--sequence", required=True)
    verify.add_argument("--bag", type=Path, required=True)

    convert = subparsers.add_parser("convert", help="Verify then run official rosbags-convert --dst-version 8.")
    convert.add_argument("--manifest", type=Path, required=True)
    convert.add_argument("--sequence", required=True)
    convert.add_argument("--src", type=Path, required=True)
    convert.add_argument("--dst", type=Path, required=True)

    inspect = subparsers.add_parser("inspect", help="Audit bag topics, rates, and MID360 per-point offsets.")
    inspect.add_argument("--bag", type=Path, required=True)
    inspect.add_argument("--report-json", type=Path)
    inspect.add_argument("--sample-lidar-messages", type=int, default=20)
    inspect.add_argument("--lidar-topic", default=MID360_LIDAR_TOPIC)
    inspect.add_argument("--imu-topic", default=MID360_IMU_TOPIC)

    download = subparsers.add_parser(
        "download",
        help="Download a manifest URL to a .part file and promote it only after SHA256 passes.",
    )
    download.add_argument("--manifest", type=Path, required=True)
    download.add_argument("--sequence", required=True)
    download.add_argument("--kind", choices=("bag", "gt"), required=True)
    download.add_argument("--output", type=Path, required=True)
    download.add_argument(
        "--accept-external-data-terms",
        action="store_true",
        help="Confirm that dataset-page terms were reviewed before downloading external data.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_entry(manifest_path: Path, sequence: str) -> dict[str, object]:
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    sequences = payload.get("sequences", {}) if isinstance(payload, dict) else {}
    if sequence not in sequences:
        raise SystemExit(f"Sequence '{sequence}' is not present in the frozen manifest.")
    return sequences[sequence]


def verify_bag(manifest_path: Path, sequence: str, bag_path: Path) -> dict[str, object]:
    entry = manifest_entry(manifest_path, sequence)
    bag_path = bag_path.expanduser().resolve()
    if not bag_path.is_file():
        raise SystemExit(f"ROS1 bag not found: {bag_path}")
    expected = str(entry.get("bag_sha256", "")).strip().lower()
    if len(expected) != 64:
        raise SystemExit(f"Manifest has no valid bag_sha256 for {sequence}.")
    actual = sha256_file(bag_path)
    passed = actual == expected
    result = {
        "sequence": sequence,
        "bag": str(bag_path),
        "size_bytes": bag_path.stat().st_size,
        "expected_sha256": expected,
        "actual_sha256": actual,
        "passed": passed,
    }
    if not passed:
        raise SystemExit(
            f"SHA256 mismatch for {sequence}: expected {expected}, actual {actual}."
        )
    return result


def inspect_ros2_bag(
    bag_path: Path, lidar_topic: str, imu_topic: str, sample_count: int
) -> dict[str, object]:
    store = build_typestore()
    with Rosbag2Reader(bag_path) as reader:
        duration_sec = reader.duration / 1.0e9
        connections = {
            connection.topic: connection for connection in reader.connections
        }
        topic_report = {
            connection.topic: {
                "msgtype": connection.msgtype,
                "message_count": connection.msgcount,
                "average_rate_hz": (
                    connection.msgcount / duration_sec if duration_sec > 0 else 0.0
                ),
            }
            for connection in reader.connections
        }
        lidar_connection = connections.get(lidar_topic)
        imu_connection = connections.get(imu_topic)
        if lidar_connection is None or imu_connection is None:
            missing = [
                topic
                for topic, connection in ((lidar_topic, lidar_connection), (imu_topic, imu_connection))
                if connection is None
            ]
            raise SystemExit(f"ROS2 bag is missing required topics: {', '.join(missing)}")

        lidar_type = lidar_connection.msgtype
        if lidar_type not in (OLD_LIVOX_TYPE, DRIVER2_LIVOX_TYPE):
            point_audit = {
                "supported": False,
                "reason": f"Unexpected MID360 message type: {lidar_type}",
            }
        else:
            sampled = 0
            point_counts: list[int] = []
            offset_max_ns: list[int] = []
            nonmonotonic_offset_messages = 0
            point_count_mismatch_messages = 0
            deserialize_errors: list[str] = []
            for connection, _, raw in reader.messages(connections=[lidar_connection]):
                if sampled >= sample_count:
                    break
                try:
                    message = store.deserialize_cdr(raw, connection.msgtype)
                except Exception as exc:  # report the exact incompatibility instead of guessing
                    deserialize_errors.append(str(exc))
                    break
                offsets = np.asarray([point.offset_time for point in message.points], dtype=np.int64)
                point_counts.append(len(offsets))
                offset_max_ns.append(int(offsets.max()) if len(offsets) else 0)
                if len(offsets) > 1 and np.any(np.diff(offsets) < 0):
                    nonmonotonic_offset_messages += 1
                if int(message.point_num) != len(offsets):
                    point_count_mismatch_messages += 1
                sampled += 1
            point_audit = {
                "supported": not deserialize_errors,
                "sampled_messages": sampled,
                "deserialize_errors": deserialize_errors,
                "point_count_min": min(point_counts) if point_counts else 0,
                "point_count_max": max(point_counts) if point_counts else 0,
                "offset_max_ns_min": min(offset_max_ns) if offset_max_ns else 0,
                "offset_max_ns_max": max(offset_max_ns) if offset_max_ns else 0,
                "nonmonotonic_offset_messages": nonmonotonic_offset_messages,
                "point_count_mismatch_messages": point_count_mismatch_messages,
                "per_point_time_offsets_preserved": bool(
                    sampled
                    and not deserialize_errors
                    and max(offset_max_ns, default=0) > 0
                    and point_count_mismatch_messages == 0
                ),
                "offset_order_note": (
                    "Livox packets need not store points in offset_time order; "
                    "nonmonotonic count is reported but is not a rejection criterion."
                ),
            }
        driver_compatibility = (
            "direct_fast_livo2"
            if lidar_type == DRIVER2_LIVOX_TYPE
            else "requires_offline_driver2_rewrite"
            if lidar_type == OLD_LIVOX_TYPE
            else "unsupported"
        )
        return {
            "bag_format": "rosbag2",
            "bag": str(bag_path),
            "start_time_ns": reader.start_time,
            "end_time_ns": reader.end_time,
            "duration_sec": duration_sec,
            "message_count": reader.message_count,
            "topics": topic_report,
            "required_topics": {"lidar": lidar_topic, "imu": imu_topic},
            "mid360_message_type": lidar_type,
            "driver_compatibility": driver_compatibility,
            "point_time_audit": point_audit,
            "passed": (
                driver_compatibility in ("direct_fast_livo2", "requires_offline_driver2_rewrite")
                and point_audit.get("per_point_time_offsets_preserved", False)
                and imu_connection.msgcount > 0
            ),
        }


def inspect_ros1_bag(bag_path: Path, lidar_topic: str, imu_topic: str) -> dict[str, object]:
    with Rosbag1Reader(bag_path) as reader:
        duration_sec = reader.duration / 1.0e9
        topics = {
            connection.topic: {
                "msgtype": connection.msgtype,
                "message_count": connection.msgcount,
                "average_rate_hz": connection.msgcount / duration_sec if duration_sec > 0 else 0.0,
            }
            for connection in reader.connections
        }
        missing = [topic for topic in (lidar_topic, imu_topic) if topic not in topics]
        return {
            "bag_format": "rosbag1",
            "bag": str(bag_path),
            "start_time_ns": reader.start_time,
            "end_time_ns": reader.end_time,
            "duration_sec": duration_sec,
            "message_count": reader.message_count,
            "topics": topics,
            "required_topics": {"lidar": lidar_topic, "imu": imu_topic},
            "missing_required_topics": missing,
            "passed": not missing,
            "next_step": "Run the official rosbags-convert wrapper, then audit per-point offsets in ROS2.",
        }


def inspect_bag(bag_path: Path, lidar_topic: str, imu_topic: str, sample_count: int) -> dict[str, object]:
    bag_path = bag_path.expanduser().resolve()
    if bag_path.is_file() and bag_path.suffix == ".bag":
        return inspect_ros1_bag(bag_path, lidar_topic, imu_topic)
    if bag_path.is_dir() and (bag_path / "metadata.yaml").is_file():
        return inspect_ros2_bag(bag_path, lidar_topic, imu_topic, sample_count)
    raise SystemExit(f"Unsupported bag path (expected ROS1 .bag or ROS2 directory): {bag_path}")


def main() -> int:
    args = parse_args()
    if args.command == "download":
        if not args.accept_external_data_terms:
            raise SystemExit(
                "Review the official external data terms, then pass "
                "--accept-external-data-terms explicitly."
            )
        manifest_path = args.manifest.expanduser().resolve()
        entry = manifest_entry(manifest_path, args.sequence)
        url_key = f"{args.kind}_download_url"
        hash_key = f"{args.kind}_sha256"
        url = str(entry.get(url_key, "")).strip()
        expected = str(entry.get(hash_key, "")).strip().lower()
        if not url:
            raise SystemExit(
                f"No {url_key} is frozen for {args.sequence}; use the official M3DGR data page "
                "and do not substitute a different sequence."
            )
        if len(expected) != 64:
            raise SystemExit(f"No valid {hash_key} is frozen for {args.sequence}.")
        output = args.output.expanduser().resolve()
        partial = output.with_name(output.name + ".part")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            actual = sha256_file(output)
            if actual != expected:
                raise SystemExit(
                    f"Existing output checksum mismatch: expected {expected}, actual {actual}."
                )
            print(f"already_verified=true output={output} sha256={actual}")
            return 0
        subprocess.run(
            [
                "curl",
                "--location",
                "--fail",
                "--show-error",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                url,
            ],
            check=True,
        )
        actual = sha256_file(partial)
        if actual != expected:
            raise SystemExit(
                f"Downloaded .part checksum mismatch: expected {expected}, actual {actual}. "
                f"The untrusted partial file was kept at {partial}."
            )
        partial.replace(output)
        print(f"download_verified=true output={output} sha256={actual}")
        return 0
    if args.command == "verify":
        result = verify_bag(
            args.manifest.expanduser().resolve(), args.sequence, args.bag
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "convert":
        manifest_path = args.manifest.expanduser().resolve()
        src = args.src.expanduser().resolve()
        dst = args.dst.expanduser().resolve()
        result = verify_bag(manifest_path, args.sequence, src)
        if dst.exists():
            raise SystemExit(f"Destination already exists; refusing to overwrite: {dst}")
        command = [
            "rosbags-convert",
            "--src",
            str(src),
            "--dst",
            str(dst),
            "--dst-version",
            "8",
        ]
        subprocess.run(command, check=True)
        result["conversion_command"] = command
        result["ros2_bag"] = str(dst)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "inspect":
        if args.sample_lidar_messages <= 0:
            raise SystemExit("--sample-lidar-messages must be positive.")
        result = inspect_bag(
            args.bag,
            args.lidar_topic,
            args.imu_topic,
            args.sample_lidar_messages,
        )
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.report_json:
            report_path = args.report_json.expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
