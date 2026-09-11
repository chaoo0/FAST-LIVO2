#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_m3dgr_runs.py"
SPEC = importlib.util.spec_from_file_location("summarize_m3dgr_runs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RotationDistancesTest(unittest.TestCase):
    def test_identical_printed_quaternions_have_zero_distance(self):
        quaternions = np.asarray(
            [
                [-0.467015, -0.007540, 0.878705, 0.098579],
                [0.000352, 0.000135, -0.000310, 1.000000],
            ]
        )
        distances = MODULE.rotation_distances_xyzw(quaternions, quaternions.copy())
        np.testing.assert_array_equal(distances, np.zeros(2))

    def test_antipodal_quaternions_are_the_same_rotation(self):
        first = np.asarray([[0.0, 0.0, 0.0, 2.0]])
        distances = MODULE.rotation_distances_xyzw(first, -first)
        np.testing.assert_array_equal(distances, np.zeros(1))

    def test_known_rotation_distance(self):
        angle = 0.25
        first = np.asarray([[0.0, 0.0, 0.0, 1.0]])
        second = np.asarray([[0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)]])
        self.assertAlmostEqual(MODULE.rotation_distances_xyzw(first, second)[0], angle, places=14)


if __name__ == "__main__":
    unittest.main()
