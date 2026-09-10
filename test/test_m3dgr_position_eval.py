#!/usr/bin/env python3

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_m3dgr_position.py"
SPEC = importlib.util.spec_from_file_location("evaluate_m3dgr_position", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PositionEvaluationTest(unittest.TestCase):
    def test_recovers_known_rigid_frame_change_without_scale(self):
        times = np.arange(0.0, 12.01, 0.2)
        gt_positions = np.column_stack((times, np.sin(times * 0.3), 0.1 * times))
        angle = 0.4
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = np.array([3.0, -2.0, 0.7])
        estimate_positions = (rotation.T @ (gt_positions - translation).T).T
        quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (len(times), 1))

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            gt_path = temporary / "gt.txt"
            estimate_path = temporary / "estimate.txt"
            np.savetxt(gt_path, np.column_stack((times, gt_positions, quaternions)))
            np.savetxt(estimate_path, np.column_stack((times, estimate_positions, quaternions)))
            result = MODULE.evaluate(gt_path, estimate_path, 0.25, [1.0, 5.0, 10.0], 0.01)

        self.assertEqual(result["metric_scope"], "position_only")
        self.assertLess(result["ate"]["rmse_m"], 1.0e-10)
        for metric in result["position_rpe"].values():
            self.assertGreater(metric["pairs"], 0)
            self.assertLess(metric["rmse_m"], 1.0e-10)
        self.assertIsNotNone(result["warning"])

    def test_duplicate_gt_timestamps_are_averaged(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "trajectory.txt"
            np.savetxt(
                path,
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0],
                        [1.0, 3.0, 0.0, 0.0],
                        [2.0, 4.0, 0.0, 0.0],
                    ]
                ),
            )
            times, positions, duplicates = MODULE.load_tum_positions(path)
        np.testing.assert_allclose(times, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(positions[:, 0], [0.0, 2.0, 4.0])
        self.assertEqual(duplicates, 1)


if __name__ == "__main__":
    unittest.main()
