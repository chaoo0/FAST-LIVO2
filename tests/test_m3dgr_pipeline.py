#!/usr/bin/env python3
"""Synthetic regression tests for the M3DGR GT preparation pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rosbags.rosbag2 import Writer
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_m3dgr_mocap_dataset import BASIC18_FEATURE_NAMES, FROZEN_SPLITS  # noqa: E402
from mamba_pose_se3 import (  # noqa: E402
    invert_transform,
    poses_from_transforms,
    se3_exp,
    se3_log,
    transform_from_pose,
)
from prepare_m3dgr_bag import (  # noqa: E402
    DRIVER2_LIVOX_TYPE,
    MID360_IMU_TOPIC,
    MID360_LIDAR_TOPIC,
    OLD_LIVOX_TYPE,
    build_typestore,
    inspect_ros2_bag,
)


def run_script(name: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *arguments],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


class Se3Tests(unittest.TestCase):
    def test_exp_log_round_trip(self) -> None:
        twists = [
            np.zeros(6),
            np.array([1.0e-9, -2.0e-9, 3.0e-9, 0.2, -0.1, 0.3]),
            np.array([0.3, -0.2, 0.1, 1.0, -0.5, 0.25]),
        ]
        for twist in twists:
            np.testing.assert_allclose(se3_log(se3_exp(twist)), twist, atol=1.0e-10)


class AlignmentTests(unittest.TestCase):
    @staticmethod
    def trajectory(relative_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.asarray(relative_time, dtype=np.float64)
        positions = np.column_stack(
            [
                0.08 * t * t + 0.15 * np.sin(1.7 * t),
                0.4 * np.sin(0.63 * t) + 0.02 * t * t,
                0.08 * np.cos(1.13 * t),
            ]
        )
        rotvec = np.column_stack(
            [
                0.08 * np.sin(0.81 * t),
                0.05 * np.cos(1.21 * t),
                0.07 * t + 0.12 * np.sin(0.47 * t),
            ]
        )
        return positions, Rotation.from_rotvec(rotvec).as_quat()

    def test_auto_time_alignment_and_trust_gate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m3dgr_align_", dir="/tmp") as temp:
            root = Path(temp)
            epoch = 1732438713.0
            gt_relative = np.arange(0.0, 8.0, 0.0025)
            gt_pos, gt_quat = self.trajectory(gt_relative)
            gt_path = root / "synthetic.tum"
            np.savetxt(
                gt_path,
                np.column_stack([epoch + gt_relative, gt_pos, gt_quat]),
                fmt="%.15g",
            )

            true_offset = 0.037
            lio_relative = np.arange(0.3, 7.6, 0.05)
            mocap_pos, mocap_quat = self.trajectory(lio_relative + true_offset)
            extrinsic = transform_from_pose(
                np.array([0.04, -0.02, 0.03]),
                Rotation.from_rotvec([0.02, -0.015, 0.01]).as_quat(),
            )
            world_align = transform_from_pose(
                np.array([1.0, -2.0, 0.4]),
                Rotation.from_rotvec([0.1, -0.05, 0.2]).as_quat(),
            )
            mocap_transforms = np.stack(
                [transform_from_pose(p, q) for p, q in zip(mocap_pos, mocap_quat)]
            )
            lio_transforms = np.stack(
                [world_align @ pose @ extrinsic for pose in mocap_transforms]
            )
            lio_pos, lio_quat = poses_from_transforms(lio_transforms)
            rows = len(lio_relative)
            frame = pd.DataFrame(
                {
                    "timestamp": epoch + lio_relative,
                    "pos_x": lio_pos[:, 0],
                    "pos_y": lio_pos[:, 1],
                    "pos_z": lio_pos[:, 2],
                    "rot_x": lio_quat[:, 0],
                    "rot_y": lio_quat[:, 1],
                    "rot_z": lio_quat[:, 2],
                    "rot_w": lio_quat[:, 3],
                    "vel_x": np.gradient(lio_pos[:, 0], 0.05),
                    "vel_y": np.gradient(lio_pos[:, 1], 0.05),
                    "vel_z": np.gradient(lio_pos[:, 2], 0.05),
                    "bias_g_x": np.zeros(rows),
                    "bias_g_y": np.zeros(rows),
                    "bias_g_z": np.zeros(rows),
                    "bias_a_x": np.zeros(rows),
                    "bias_a_y": np.zeros(rows),
                    "bias_a_z": np.zeros(rows),
                    "effective_feature_num": np.full(rows, 100),
                    "avg_residual": np.full(rows, 0.01),
                    "history_size": np.full(rows, 10),
                    "ready_flag": np.ones(rows),
                    "gt_pos_x": np.nan,
                    "gt_pos_y": np.nan,
                    "gt_pos_z": np.nan,
                    "gt_rot_x": np.nan,
                    "gt_rot_y": np.nan,
                    "gt_rot_z": np.nan,
                    "gt_rot_w": np.nan,
                }
            )
            lio_csv = root / "lio.csv"
            aligned_csv = root / "aligned.csv"
            report_json = root / "alignment.json"
            frame.to_csv(lio_csv, index=False)
            extrinsic_pos, extrinsic_quat = poses_from_transforms(extrinsic)
            run_script(
                "align_m3dgr_mocap_gt.py",
                "--lio-csv", str(lio_csv),
                "--mocap-tum", str(gt_path),
                "--output-csv", str(aligned_csv),
                "--report-json", str(report_json),
                "--run-id", "Varying-illu02",
                "--gt-body-to-imu",
                *[str(value) for value in np.concatenate([extrinsic_pos, extrinsic_quat])],
                "--rigid-body-frame-status", "confirmed",
                "--extrinsic-source", "synthetic_test",
            )
            report = json.loads(report_json.read_text())
            self.assertEqual(report["status"], "trusted_gt")
            self.assertGreaterEqual(report["interpolation"]["match_ratio"], 0.95)
            self.assertAlmostEqual(
                report["time_alignment"]["offset_sec"], true_offset, delta=0.003
            )
            self.assertLess(
                report["initial_alignment_residual"]["translation_m"]["median"], 0.02
            )
            aligned = pd.read_csv(aligned_csv)
            self.assertEqual(set(aligned.loc[aligned.gt_valid == 1, "gt_status"]), {"trusted_gt"})

    def test_multirun_handeye_consistency(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m3dgr_handeye_", dir="/tmp") as temp:
            root = Path(temp)
            extrinsic = transform_from_pose(
                np.array([0.07, -0.03, 0.04]),
                Rotation.from_rotvec([0.08, -0.04, 0.05]).as_quat(),
            )
            manifest_runs = []
            for run_index, run_id in enumerate(("Dynamic01", "Occlusion01")):
                gt_relative = np.arange(0.0, 10.0, 0.01)
                gt_pos, gt_quat = self.trajectory(gt_relative + run_index * 0.23)
                gt_path = root / f"{run_id}.tum"
                np.savetxt(
                    gt_path,
                    np.column_stack([1000.0 + gt_relative, gt_pos, gt_quat]),
                    fmt="%.15g",
                )
                lio_relative = np.arange(0.2, 9.8, 0.05)
                mocap_pos, mocap_quat = self.trajectory(lio_relative + run_index * 0.23)
                world_align = transform_from_pose(
                    np.array([run_index, -0.5 * run_index, 0.2]),
                    Rotation.from_rotvec([0.03 * run_index, 0.02, -0.1]).as_quat(),
                )
                mocap = np.stack(
                    [transform_from_pose(p, q) for p, q in zip(mocap_pos, mocap_quat)]
                )
                lio = np.stack([world_align @ pose @ extrinsic for pose in mocap])
                lio_pos, lio_quat = poses_from_transforms(lio)
                rows = len(lio_relative)
                frame = pd.DataFrame(
                    {
                        "timestamp": 1000.0 + lio_relative,
                        "pos_x": lio_pos[:, 0], "pos_y": lio_pos[:, 1], "pos_z": lio_pos[:, 2],
                        "rot_x": lio_quat[:, 0], "rot_y": lio_quat[:, 1],
                        "rot_z": lio_quat[:, 2], "rot_w": lio_quat[:, 3],
                        "vel_x": np.zeros(rows), "vel_y": np.zeros(rows), "vel_z": np.zeros(rows),
                        "bias_g_x": np.zeros(rows), "bias_g_y": np.zeros(rows), "bias_g_z": np.zeros(rows),
                        "bias_a_x": np.zeros(rows), "bias_a_y": np.zeros(rows), "bias_a_z": np.zeros(rows),
                        "effective_feature_num": np.full(rows, 100), "avg_residual": np.full(rows, 0.01),
                        "history_size": np.full(rows, 10), "ready_flag": np.ones(rows),
                        "gt_pos_x": np.nan, "gt_pos_y": np.nan, "gt_pos_z": np.nan,
                        "gt_rot_x": np.nan, "gt_rot_y": np.nan, "gt_rot_z": np.nan, "gt_rot_w": np.nan,
                    }
                )
                csv_path = root / f"{run_id}.csv"
                frame.to_csv(csv_path, index=False)
                manifest_runs.append(
                    {
                        "run_id": run_id,
                        "lio_csv": str(csv_path),
                        "mocap_tum": str(gt_path),
                        "time_offset_sec": 0.0,
                    }
                )
            manifest = root / "handeye.yaml"
            output = root / "handeye.json"
            manifest.write_text(yaml.safe_dump({"runs": manifest_runs}, sort_keys=False))
            run_script(
                "estimate_m3dgr_handeye.py",
                "--manifest", str(manifest),
                "--output-json", str(output),
                "--relative-stride", "2",
                "--min-rotation-deg", "0.1",
                "--min-relative-motions-per-run", "10",
            )
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "handeye_consistent_unconfirmed")
            estimated = np.asarray(report["shared_extrinsic"]["matrix"])
            error = se3_log(invert_transform(extrinsic) @ estimated)
            self.assertLess(np.linalg.norm(error[:3]), np.deg2rad(0.2))
            self.assertLess(np.linalg.norm(error[3:]), 0.005)


class DatasetTests(unittest.TestCase):
    @staticmethod
    def make_aligned_run(path: Path, run_id: str, phase: float) -> None:
        timestamps = 1000.0 + np.arange(60) * 0.05
        rows = len(timestamps)
        est_pos = np.column_stack(
            [0.05 * np.arange(rows), 0.1 * np.sin(phase + np.arange(rows) * 0.1), np.zeros(rows)]
        )
        est_quat = Rotation.from_euler("z", 0.01 * np.arange(rows)).as_quat()
        est = np.stack([transform_from_pose(p, q) for p, q in zip(est_pos, est_quat)])
        twists = np.column_stack(
            [
                0.01 * np.sin(np.arange(rows) * 0.1 + phase),
                np.zeros(rows),
                np.zeros(rows),
                0.25 + 0.05 * np.sin(np.arange(rows) * 0.2 + phase),
                np.zeros(rows),
                np.zeros(rows),
            ]
        )
        ref = np.stack([pose @ se3_exp(twist) for pose, twist in zip(est, twists)])
        ref_pos, ref_quat = poses_from_transforms(ref)
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "pos_x": est_pos[:, 0], "pos_y": est_pos[:, 1], "pos_z": est_pos[:, 2],
                "rot_x": est_quat[:, 0], "rot_y": est_quat[:, 1],
                "rot_z": est_quat[:, 2], "rot_w": est_quat[:, 3],
                "vel_x": np.full(rows, 1.0 + phase), "vel_y": np.zeros(rows), "vel_z": np.zeros(rows),
                "bias_g_x": np.zeros(rows), "bias_g_y": np.zeros(rows), "bias_g_z": np.zeros(rows),
                "bias_a_x": np.zeros(rows), "bias_a_y": np.zeros(rows), "bias_a_z": np.zeros(rows),
                "effective_feature_num": np.full(rows, 100 + int(phase)),
                "avg_residual": np.full(rows, 0.01),
                "history_size": np.full(rows, 10), "ready_flag": np.ones(rows),
                "gt_pos_x": ref_pos[:, 0], "gt_pos_y": ref_pos[:, 1], "gt_pos_z": ref_pos[:, 2],
                "gt_rot_x": ref_quat[:, 0], "gt_rot_y": ref_quat[:, 1],
                "gt_rot_z": ref_quat[:, 2], "gt_rot_w": ref_quat[:, 3],
                "gt_valid": np.ones(rows, dtype=np.int8),
                "gt_match_bracket_sec": np.zeros(rows),
                "gt_type": np.full(rows, "mocap"),
                "gt_status": np.full(rows, "trusted_gt"),
                "run_id": np.full(rows, run_id),
                "segment_id": np.zeros(rows, dtype=np.int64),
                "gt_time_offset_sec": np.full(rows, 0.012),
                "gt_frame_convention_version": np.full(rows, "synthetic_v1"),
                "gt_extrinsic_sha256": np.full(rows, "a" * 64),
            }
        )
        frame.to_csv(path, index=False)

    def test_formal_split_train_only_normalization_and_large_label_preservation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m3dgr_dataset_", dir="/tmp") as temp:
            root = Path(temp)
            sequences: dict[str, dict[str, str]] = {}
            all_names = [name for split in ("train", "val", "test") for name in FROZEN_SPLITS[split]]
            for index, run_id in enumerate(all_names):
                csv_path = root / f"{run_id}.csv"
                report_path = root / f"{run_id}.json"
                self.make_aligned_run(csv_path, run_id, index * 0.1)
                report_path.write_text(json.dumps({"run_id": run_id, "status": "trusted_gt"}))
                split = next(split for split, names in FROZEN_SPLITS.items() if run_id in names)
                sequences[run_id] = {
                    "split": split,
                    "aligned_csv": str(csv_path),
                    "alignment_report": str(report_path),
                }
            manifest = root / "manifest.yaml"
            manifest.write_text(yaml.safe_dump({"sequences": sequences}, sort_keys=False))
            output = root / "output"
            run_script(
                "build_m3dgr_mocap_dataset.py",
                "--manifest", str(manifest),
                "--output-dir", str(output),
                "--mode", "formal",
            )
            norm = np.load(output / "m3dgr_feature_norm_train_T10.npz")
            self.assertEqual(set(norm["source_run_ids"].tolist()), set(FROZEN_SPLITS["train"]))
            for split, expected_runs in FROZEN_SPLITS.items():
                dataset = np.load(output / f"m3dgr_mocap_{split}_T10.npz")
                self.assertEqual(set(dataset["run_ids"].tolist()), set(expected_runs))
                self.assertFalse(bool(dataset["runtime_safety_filter_applied"]))
                self.assertGreater(float(np.max(np.abs(dataset["y"][:, 3]))), 0.2)
                self.assertTrue(np.isfinite(dataset["X_normalized"]).all())

    def test_covariance_profile_selection_uses_all_train_runs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m3dgr_profiles_", dir="/tmp") as temp:
            root = Path(temp)
            arguments: list[str] = []
            for index, run_id in enumerate(FROZEN_SPLITS["train"]):
                current_path = root / f"current_{run_id}.csv"
                candidate_path = root / f"candidate_{run_id}.csv"
                self.make_aligned_run(current_path, run_id, index * 0.1)
                candidate = pd.read_csv(current_path)
                candidate["pos_x"] = candidate["pos_x"] + 0.5
                candidate_path.write_text(candidate.to_csv(index=False))
                arguments.extend(
                    ["--trajectory", f"current:{run_id}:{current_path}"]
                )
                arguments.extend(
                    ["--trajectory", f"kalibr_variance:{run_id}:{candidate_path}"]
                )
            report = root / "comparison.json"
            frozen = root / "frozen.yaml"
            run_script(
                "evaluate_m3dgr_lio_profiles.py",
                *arguments,
                "--output-json", str(report),
                "--freeze-output-yaml", str(frozen),
            )
            payload = json.loads(report.read_text())
            self.assertEqual(payload["selected_profile"], "current")
            freeze_payload = yaml.safe_load(frozen.read_text())
            self.assertFalse(freeze_payload["validation_or_test_used_for_selection"])


class BagRewriteTests(unittest.TestCase):
    def test_old_livox_cdr_rewrite_preserves_offsets(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m3dgr_bag_", dir="/tmp") as temp:
            root = Path(temp)
            source = root / "old_bag"
            output = root / "driver2_bag"
            report = root / "rewrite.json"
            store = build_typestore()
            types = store.types
            time_type = types["builtin_interfaces/msg/Time"]
            header_type = types["std_msgs/msg/Header"]
            point_type = types["livox_ros_driver/msg/CustomPoint"]
            custom_type = types[OLD_LIVOX_TYPE]
            quaternion_type = types["geometry_msgs/msg/Quaternion"]
            vector_type = types["geometry_msgs/msg/Vector3"]
            imu_type = types["sensor_msgs/msg/Imu"]
            header = header_type(time_type(1, 0), "mid360")
            points = [
                point_type(offset, float(index), 0.0, 0.0, 1, 0, 1)
                for index, offset in enumerate((0, 20_000_000, 10_000_000, 90_000_000))
            ]
            lidar_message = custom_type(
                header, 1_000_000_000, len(points), 1, np.zeros(3, dtype=np.uint8), points
            )
            imu_message = imu_type(
                header,
                quaternion_type(0.0, 0.0, 0.0, 1.0),
                np.zeros(9, dtype=np.float64),
                vector_type(0.0, 0.0, 0.0),
                np.zeros(9, dtype=np.float64),
                vector_type(0.0, 0.0, 9.81),
                np.zeros(9, dtype=np.float64),
            )
            with Writer(source, version=8) as writer:
                lidar_connection = writer.add_connection(
                    MID360_LIDAR_TOPIC, OLD_LIVOX_TYPE, typestore=store
                )
                imu_connection = writer.add_connection(
                    MID360_IMU_TOPIC, "sensor_msgs/msg/Imu", typestore=store
                )
                for index in range(3):
                    stamp = 1_000_000_000 + index * 100_000_000
                    writer.write(
                        lidar_connection,
                        stamp,
                        store.serialize_cdr(lidar_message, OLD_LIVOX_TYPE),
                    )
                    writer.write(
                        imu_connection,
                        stamp + 1,
                        store.serialize_cdr(imu_message, "sensor_msgs/msg/Imu"),
                    )
            before = inspect_ros2_bag(source, MID360_LIDAR_TOPIC, MID360_IMU_TOPIC, 3)
            self.assertEqual(before["driver_compatibility"], "requires_offline_driver2_rewrite")
            self.assertTrue(before["point_time_audit"]["per_point_time_offsets_preserved"])
            run_script(
                "rewrite_m3dgr_livox_driver2_bag.py",
                "--input-bag", str(source),
                "--output-bag", str(output),
                "--report-json", str(report),
                "--sample-lidar-messages", "3",
            )
            after = inspect_ros2_bag(output, MID360_LIDAR_TOPIC, MID360_IMU_TOPIC, 3)
            self.assertEqual(after["mid360_message_type"], DRIVER2_LIVOX_TYPE)
            self.assertTrue(after["point_time_audit"]["per_point_time_offsets_preserved"])
            self.assertTrue(json.loads(report.read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
