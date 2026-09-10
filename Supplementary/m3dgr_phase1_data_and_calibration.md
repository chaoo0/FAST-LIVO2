# M3DGR Phase 1: data and calibration audit

Date: 2026-09-10

Development baseline: `d6e22ad4e1ee478e41050cfa83a3c35e98d00ee5`

M3DGR reference snapshot: `sjtuyinjie/M3DGR@e0cf7d59c9a5a3df515624034698d976abc26549`

## Evidence scope

- 【事实】Primary dataset references are the official [M3DGR repository](https://github.com/sjtuyinjie/M3DGR), its [calibration document](https://github.com/sjtuyinjie/M3DGR/blob/main/calibration.md), and its [FAST-LIVO2 adaptation](https://github.com/sjtuyinjie/M3DGR/tree/main/baseline_systems/Fast_LIVO2_M3DGR). The snapshot above was frozen locally for this audit.
- 【事实】The official documentation identifies Livox MID-360 as a 10 Hz non-repetitive LiDAR with a built-in 200 Hz six-axis IMU; D435i RGB is 640 x 480. The published ROS topics are `/livox/mid360/lidar`, `/livox/mid360/imu`, and `/camera/color/image_raw/compressed`.
- 【事实】The dataset has no hardware trigger across sensors; the official documentation states that software synchronization is used.
- 【未知】The public documentation does not specify a single numerical residual synchronization accuracy for the sequences inspected here. A zero time offset is therefore not being claimed as calibrated truth.

## Converted ROS 2 bags

Both `ros2 bag info` and `rosbag2_py.SequentialReader` were used. The table reports storage timestamps and metadata counts. All serialized records in every converted bag were traversed. IMU, compressed image, and odometry messages were fully deserialized in the normal audit; MID-360 payload deserialization was sampled because Python deserialization is expensive. Corridor02 was additionally checked with full deserialization of all selected topics.

| Local sequence | Duration (s) | All messages | MID-360 LiDAR | MID-360 IMU | RGB compressed | Wheel odom | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Corridor01 | 383.392858 | 275,364 | 3,834 | 76,679 | 11,494 | 7,668 | readable |
| Corridor02 | 293.356221 | 210,711 | 2,934 | 58,672 | 8,795 | 5,867 | fully deserialized |
| Outdoor01 | 411.566921 | 314,576 | 4,116 | 82,312 | 12,338 | 8,231 | readable |
| Outdoor010 | 262.563796 | 200,677 | 2,626 | 52,512 | 7,871 | 5,251 | readable, partial copy |
| Outdoor04 | 782.815876 | 598,231 | 7,828 | 156,583 | 23,470 | 15,658 | readable |

- 【事实】Corridor02 full check deserialized 2,934 `livox_ros_driver2/msg/CustomMsg`, 58,672 `sensor_msgs/msg/Imu`, 8,795 `sensor_msgs/msg/CompressedImage`, and 5,867 `nav_msgs/msg/Odometry` messages with zero failures and zero header timestamp inversions.
- 【事实】Corridor02 observed mean header rates were approximately 10.000 Hz LiDAR, 200.000 Hz IMU, 29.977 Hz compressed RGB, and 20.001 Hz odometry. MID-360 line IDs were 0 to 3 and sampled scan offsets extended to about 100.51 ms.
- 【事实】`livox_ros_driver2` ROS 2 interfaces are installed and loadable. The legacy ROS 1 `livox_ros_driver` interface and `gnss_comm` are not installed, so converted Avia and some GNSS topics cannot currently be deserialized by native ROS 2 tools. They are outside this MID-360 LIO input path.
- 【事实】`Outdoor010` is a partial recovery of `Outdoor01`, not an independent official sequence. It is excluded from experimental splits.

The reproducible checker is `scripts/audit_m3dgr_bag.py`; `--full-lidar-deserialize` requests complete MID-360 payload deserialization.

## Transform convention and configured extrinsics

The official calibration convention is

```text
p_target = R_target_source * p_source + t_target_source
```

with translation in metres.

| Transform | Translation (m) | Rotation |
|---|---|---|
| MID-360 -> built-in IMU | `[-0.011, -0.02329, 0.04412]` | identity |
| MID-360 -> D435i RGB | `[-0.0442358, -0.411712, 0.168568]` | `[0.0722207,-0.997387,-0.0020542; 0.521005,0.039482,-0.85264; 0.850493,0.060508,0.522494]` |

- 【事实】The ROS 2 baseline passes `extrinsic_R/T` directly into the LiDAR-to-IMU point transform and internally inverts it where the inverse IMU-to-LiDAR transform is required. The configured direction therefore matches the official calibration convention.
- 【事实】The visual code applies `p_camera = Rcl * p_lidar + Pcl`, so the stored camera extrinsic direction also matches the published MID-360-to-camera transform.
- 【事实】Visual processing is disabled in Phase 1 (`common.img_en=0`), so camera calibration is recorded but not experimentally validated by the LIO runs.

## Camera model

The recorded future LIVO configuration uses the official D435i RGB pinhole model:

| Item | Value |
|---|---|
| Resolution | 640 x 480 |
| `fx`, `fy` | 607.79772949218, 607.83526613281 |
| `cx`, `cy` | 328.79772949218, 245.53321838378 |
| `k1`, `k2`, `p1`, `p2` | all zero in the supplied model |

【未知】The official dataset configuration uses an image offset of 0.1 s, but Phase 1 did not independently estimate it. It has no effect with images disabled and must be revalidated before LIVO.

## LIO estimator parameters

`config/m3dgr_mid360_lio.yaml` follows the official M3DGR FAST-LIVO2 adaptation for the input topics and the main MID-360 preprocessing choices:

| Parameter | Value | Meaning/source status |
|---|---:|---|
| `scan_line` | 4 | consistent with observed line IDs 0..3 |
| `scan_rate` | 10 Hz | official sensor specification and observed data |
| `blind` | 0.8 m | official adapted FAST-LIVO2 tuning |
| `point_filter_num` | 1 | official adapted FAST-LIVO2 tuning |
| `filter_size_surf` | 0.1 m | official adapted FAST-LIVO2 tuning |
| `acc_cov` | 0.5 | official adapted FAST-LIVO2 process-noise scale |
| `gyr_cov` | 0.3 | official adapted FAST-LIVO2 process-noise scale |
| voxel size | 0.5 m | ROS 2 baseline/default map tuning |

【事实】The published Allan parameters have physical noise-density/random-walk units, while the baseline code injects `acc_cov` and `gyr_cov` as diagonal process scales multiplied by `dt^2`. Substituting the Allan numbers directly would change the estimator model and would not be a configuration-only equivalence. Phase 1 therefore retains the official adapted FAST-LIVO2 tuning and documents that it is not a manufacturer noise calibration.

【事实】The ROS 2 baseline declares and reads `lio.min_iterations`, although the upstream-style configuration names the setting `max_iterations`. Both keys are set to 5 to preserve baseline behavior. Correcting the key in source is a Phase 2 code-review item.

## Ground truth audit

| Sequence | Local GT form | Resulting evaluation scope |
|---|---|---|
| Outdoor01 | TUM-like time and position; all quaternions `[0,0,0,1]` | position only |
| Outdoor04 | TUM-like time and position; all quaternions `[0,0,0,1]` | position only |
| Corridor01/02 | one terminal ArUco rotation/translation plus bag duration | terminal transform, not a time trajectory |

- 【事实】Outdoor01 has 6,174 rows but only 2,861 unique timestamps; 3,313 duplicate timestamps are merged by mean position. Outdoor04 has 11,745 rows, 6,798 unique timestamps, and 4,947 duplicates.
- 【事实】Every Outdoor01 and Outdoor04 quaternion in the available files is identity. Rotational ATE/RPE and full SE(3) trajectory accuracy cannot be evaluated from these files.
- 【推断】The outdoor position is an RTK-derived reference, while FAST-LIVO2 publishes the IMU-state position. Because the GT provides no time-varying orientation, the published antenna-to-LiDAR/IMU lever arm cannot be rotated into the evaluation frame at every timestamp. The reported position error may therefore contain a reference-point-dependent component.
- 【未知】The exact RTK processing pipeline, duplicate-row cause, and time-varying reference orientation are not supplied in the local artifacts. These are blockers for a rigorous six-degree-of-freedom GT claim.

## Phase 1 data conclusion

【事实】The MID-360 and IMU data needed for LIO are readable and were exercised end to end. 【观点】They are suitable for a position-only Phase 1 smoke/baseline evaluation, but the currently available M3DGR GT artifacts are insufficient by themselves for the later local 6D pose-correction labels required by the Mamba research objective.
