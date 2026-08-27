#!/usr/bin/env python3
"""Shared, convention-explicit SE(3) helpers for MambaPose offline tools.

Transform convention
--------------------
``T_A_B`` maps coordinates expressed in frame B into frame A.  Pose
quaternions use ROS/TUM ``[x, y, z, w]`` order.  A right correction is
therefore ``T_ref = T_est @ Exp(xi)`` and ``xi = Log(inv(T_est) @ T_ref)``.
The twist layout used by this project is ``[phi_x, phi_y, phi_z,
rho_x, rho_y, rho_z]``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


TWIST_NAMES = (
    "delta_theta_x",
    "delta_theta_y",
    "delta_theta_z",
    "delta_rho_x",
    "delta_rho_y",
    "delta_rho_z",
)


def skew(vector: np.ndarray) -> np.ndarray:
    """Return the 3x3 cross-product matrix for one three-vector."""

    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def normalize_quaternions(quaternions_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize finite quaternions and return normalized values plus raw norms."""

    quaternions = np.asarray(quaternions_xyzw, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"Expected quaternion array [N,4], got {quaternions.shape}.")
    if not np.isfinite(quaternions).all():
        raise ValueError("Quaternion array contains NaN or Inf.")
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("Quaternion array contains a zero-norm quaternion.")
    return quaternions / norms[:, None], norms


def transform_from_pose(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Create one homogeneous transform from position and quaternion."""

    position = np.asarray(position, dtype=np.float64).reshape(3)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(1, 4)
    quaternion, _ = normalize_quaternions(quaternion)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion[0]).as_matrix()
    transform[:3, 3] = position
    return transform


def transforms_from_poses(
    positions: np.ndarray, quaternions_xyzw: np.ndarray
) -> np.ndarray:
    """Create homogeneous transforms for arrays shaped [N,3] and [N,4]."""

    positions = np.asarray(positions, dtype=np.float64)
    quaternions, _ = normalize_quaternions(quaternions_xyzw)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected position array [N,3], got {positions.shape}.")
    if positions.shape[0] != quaternions.shape[0]:
        raise ValueError("Position and quaternion row counts differ.")
    if not np.isfinite(positions).all():
        raise ValueError("Position array contains NaN or Inf.")
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(positions), axis=0)
    transforms[:, :3, :3] = Rotation.from_quat(quaternions).as_matrix()
    transforms[:, :3, 3] = positions
    return transforms


def poses_from_transforms(transforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return positions and xyzw quaternions from [N,4,4] transforms."""

    transforms = np.asarray(transforms, dtype=np.float64)
    single = transforms.ndim == 2
    transforms = transforms[None, ...] if single else transforms
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(f"Expected transform array [N,4,4], got {transforms.shape}.")
    positions = transforms[:, :3, 3].copy()
    quaternions = Rotation.from_matrix(transforms[:, :3, :3]).as_quat()
    if single:
        return positions[0], quaternions[0]
    return positions, quaternions


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert one homogeneous rigid transform without a generic matrix inverse."""

    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def invert_transforms(transforms: np.ndarray) -> np.ndarray:
    """Vectorized rigid-transform inverse for [N,4,4]."""

    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(f"Expected transform array [N,4,4], got {transforms.shape}.")
    result = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(transforms), axis=0)
    result[:, :3, :3] = np.transpose(transforms[:, :3, :3], (0, 2, 1))
    result[:, :3, 3] = -np.einsum(
        "nij,nj->ni", result[:, :3, :3], transforms[:, :3, 3]
    )
    return result


def se3_exp(twist: np.ndarray) -> np.ndarray:
    """SE(3) exponential for [rotation-vector, local-rho] ordering."""

    twist = np.asarray(twist, dtype=np.float64).reshape(6)
    phi = twist[:3]
    rho = twist[3:]
    theta = float(np.linalg.norm(phi))
    omega = skew(phi)
    omega2 = omega @ omega
    if theta < 1.0e-8:
        v_matrix = np.eye(3) + 0.5 * omega + (1.0 / 6.0) * omega2
    else:
        theta2 = theta * theta
        v_matrix = (
            np.eye(3)
            + ((1.0 - np.cos(theta)) / theta2) * omega
            + ((theta - np.sin(theta)) / (theta2 * theta)) * omega2
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(phi).as_matrix()
    transform[:3, 3] = v_matrix @ rho
    return transform


def se3_log(transform: np.ndarray) -> np.ndarray:
    """SE(3) logarithm in [rotation-vector, local-rho] ordering."""

    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    phi = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    theta = float(np.linalg.norm(phi))
    omega = skew(phi)
    omega2 = omega @ omega
    if theta < 1.0e-8:
        v_inverse = np.eye(3) - 0.5 * omega + (1.0 / 12.0) * omega2
    else:
        coefficient = (
            1.0 / (theta * theta)
            - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
        )
        v_inverse = np.eye(3) - 0.5 * omega + coefficient * omega2
    rho = v_inverse @ transform[:3, 3]
    return np.concatenate([phi, rho])


def se3_logs(transforms: np.ndarray) -> np.ndarray:
    """Apply :func:`se3_log` to a transform array."""

    transforms = np.asarray(transforms, dtype=np.float64)
    return np.stack([se3_log(transform) for transform in transforms], axis=0)


def quaternion_angular_distance_rad(
    first_xyzw: np.ndarray, second_xyzw: np.ndarray
) -> np.ndarray:
    """Shortest quaternion angular distance in radians."""

    first, _ = normalize_quaternions(np.atleast_2d(first_xyzw))
    second, _ = normalize_quaternions(np.atleast_2d(second_xyzw))
    if first.shape != second.shape:
        raise ValueError("Quaternion distance inputs must have identical shapes.")
    dots = np.abs(np.sum(first * second, axis=1))
    return 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))


def interpolate_poses(
    source_timestamps: np.ndarray,
    source_positions: np.ndarray,
    source_quaternions_xyzw: np.ndarray,
    query_timestamps: np.ndarray,
    max_bracket_gap_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate positions and SLERP rotations at query times.

    A query is valid only when it lies inside the source range and its two
    bracketing source samples are no farther apart than ``max_bracket_gap_sec``.
    Exact timestamp matches have zero bracket gap.
    """

    source_timestamps = np.asarray(source_timestamps, dtype=np.float64)
    source_positions = np.asarray(source_positions, dtype=np.float64)
    source_quaternions, _ = normalize_quaternions(source_quaternions_xyzw)
    query_timestamps = np.asarray(query_timestamps, dtype=np.float64)
    if len(source_timestamps) < 2:
        raise ValueError("At least two source poses are required for interpolation.")
    if not np.all(np.diff(source_timestamps) > 0.0):
        raise ValueError("Source timestamps must be strictly increasing.")

    positions = np.full((len(query_timestamps), 3), np.nan, dtype=np.float64)
    quaternions = np.full((len(query_timestamps), 4), np.nan, dtype=np.float64)
    bracket_gaps = np.full(len(query_timestamps), np.inf, dtype=np.float64)
    valid = np.zeros(len(query_timestamps), dtype=bool)

    right = np.searchsorted(source_timestamps, query_timestamps, side="left")
    exact = (right < len(source_timestamps)) & np.isclose(
        source_timestamps[np.minimum(right, len(source_timestamps) - 1)],
        query_timestamps,
        rtol=0.0,
        atol=1.0e-9,
    )
    exact_indices = right[exact]
    positions[exact] = source_positions[exact_indices]
    quaternions[exact] = source_quaternions[exact_indices]
    bracket_gaps[exact] = 0.0
    valid[exact] = True

    interpolate_mask = (~exact) & (right > 0) & (right < len(source_timestamps))
    query_indices = np.flatnonzero(interpolate_mask)
    if query_indices.size:
        right_indices = right[query_indices]
        left_indices = right_indices - 1
        gaps = source_timestamps[right_indices] - source_timestamps[left_indices]
        usable = np.isfinite(gaps) & (gaps > 0.0) & (gaps <= max_bracket_gap_sec)
        usable_query = query_indices[usable]
        left_usable = left_indices[usable]
        right_usable = right_indices[usable]
        usable_gaps = gaps[usable]
        bracket_gaps[query_indices] = gaps
        if usable_query.size:
            alpha = (
                query_timestamps[usable_query] - source_timestamps[left_usable]
            ) / usable_gaps
            positions[usable_query] = (
                (1.0 - alpha[:, None]) * source_positions[left_usable]
                + alpha[:, None] * source_positions[right_usable]
            )
            for output_index, left_index, right_index, query_time in zip(
                usable_query, left_usable, right_usable, query_timestamps[usable_query]
            ):
                slerp = Slerp(
                    source_timestamps[[left_index, right_index]],
                    Rotation.from_quat(source_quaternions[[left_index, right_index]]),
                )
                quaternions[output_index] = slerp([query_time]).as_quat()[0]
            valid[usable_query] = True
    return positions, quaternions, valid, bracket_gaps


def pose_speed_profile(
    timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return midpoint timestamps, linear speed, and angular speed magnitudes."""

    timestamps = np.asarray(timestamps, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    quaternions, _ = normalize_quaternions(quaternions_xyzw)
    dt = np.diff(timestamps)
    valid = np.isfinite(dt) & (dt > 1.0e-6)
    midpoint = 0.5 * (timestamps[:-1] + timestamps[1:])
    linear = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.maximum(dt, 1.0e-12)
    relative = Rotation.from_quat(quaternions[:-1]).inv() * Rotation.from_quat(
        quaternions[1:]
    )
    angular = np.linalg.norm(relative.as_rotvec(), axis=1) / np.maximum(dt, 1.0e-12)
    valid &= np.isfinite(linear) & np.isfinite(angular)
    return midpoint[valid], linear[valid], angular[valid]


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if len(first) < 8 or np.std(first) < 1.0e-10 or np.std(second) < 1.0e-10:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def estimate_time_offset(
    lio_timestamps: np.ndarray,
    lio_positions: np.ndarray,
    lio_quaternions_xyzw: np.ndarray,
    gt_timestamps: np.ndarray,
    gt_positions: np.ndarray,
    gt_quaternions_xyzw: np.ndarray,
    search_half_range_sec: float = 0.2,
    step_sec: float = 0.001,
) -> dict[str, object]:
    """Estimate ``gt_query_time = lio_time + offset`` using speed correlation."""

    lio_mid, lio_linear, lio_angular = pose_speed_profile(
        lio_timestamps, lio_positions, lio_quaternions_xyzw
    )
    gt_mid, gt_linear, gt_angular = pose_speed_profile(
        gt_timestamps, gt_positions, gt_quaternions_xyzw
    )
    offsets = np.arange(
        -search_half_range_sec,
        search_half_range_sec + 0.5 * step_sec,
        step_sec,
        dtype=np.float64,
    )
    rows: list[tuple[float, float, float, float, int]] = []
    for offset in offsets:
        query = lio_mid + offset
        overlap = (query >= gt_mid[0]) & (query <= gt_mid[-1])
        if int(overlap.sum()) < 8:
            rows.append((float(offset), float("nan"), float("nan"), float("nan"), int(overlap.sum())))
            continue
        gt_linear_interp = np.interp(query[overlap], gt_mid, gt_linear)
        gt_angular_interp = np.interp(query[overlap], gt_mid, gt_angular)
        linear_corr = _pearson(lio_linear[overlap], gt_linear_interp)
        angular_corr = _pearson(lio_angular[overlap], gt_angular_interp)
        finite_scores = [score for score in (linear_corr, angular_corr) if np.isfinite(score)]
        score = float(np.mean(finite_scores)) if finite_scores else float("nan")
        rows.append((float(offset), score, linear_corr, angular_corr, int(overlap.sum())))
    finite_rows = [row for row in rows if np.isfinite(row[1])]
    if not finite_rows:
        raise ValueError("Time-offset correlation is undefined; the trajectories lack usable motion.")
    best = max(finite_rows, key=lambda row: row[1])
    sorted_scores = sorted((row[1] for row in finite_rows), reverse=True)
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else float("nan")
    return {
        "offset_sec": best[0],
        "score": best[1],
        "linear_correlation": best[2],
        "angular_correlation": best[3],
        "overlap_count": best[4],
        "score_margin_to_second_grid_point": best[1] - second_score,
        "search_half_range_sec": search_half_range_sec,
        "step_sec": step_sec,
        "candidate_count": len(rows),
    }


def average_transforms(transforms: np.ndarray) -> np.ndarray:
    """Average rigid transforms using a rotation mean and translation median."""

    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4) or not len(transforms):
        raise ValueError("Expected at least one transform with shape [N,4,4].")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_matrix(transforms[:, :3, :3]).mean().as_matrix()
    result[:3, 3] = np.median(transforms[:, :3, 3], axis=0)
    return result


def canonical_transform_hash(transform: np.ndarray, convention_version: str) -> str:
    """Hash a transform and its semantic convention for provenance."""

    payload = {
        "convention_version": convention_version,
        "matrix_row_major": np.asarray(transform, dtype=np.float64).reshape(4, 4).round(15).tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def matrix_to_list(transform: np.ndarray) -> list[list[float]]:
    return np.asarray(transform, dtype=np.float64).reshape(4, 4).tolist()


def percentile_summary(values: Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(values), dtype=np.float64)
    if not values.size:
        return {key: float("nan") for key in ("median", "p95", "p99", "max")}
    return {
        "median": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }
