# Phase 2: HKU reference versus chaoo0 ROS 2 audit

Date started: 2026-09-10

Development branch: `phase2/hku-ros2-audit`

Development base: `f927356db269a71b3c5f9262061303bc84adab62`, whose estimator source is still the chaoo0 ROS 2 baseline `d6e22ad4e1ee478e41050cfa83a3c35e98d00ee5`.

Official reference: frozen source directory named for `hku-mars/FAST-LIVO2@0d2c0346107b75b59934975adec9a6eeeb913c64`. The extracted directory has no `.git` metadata; its deterministic file-hash manifest digest on this host is `52b815574ef2617efa19131f32930a071ef6faf5d242252ff40cf0cda1ac7734`.

## Evidence rules

Every item is kept separate until it has a source location, reference comparison, trigger, minimal reproduction, impact, fix, and regression result. “Builds” and “runs” are not treated as proof of correctness. A fixed item remains pending until the full M3DGR regression is complete.

## Audit ledger

| ID | Classification | Status | Summary |
|---|---|---|---|
| A-001 | upstream build portability defect | fixed, full Outdoor01 regression passed | x86 `-march=native` creates an Eigen/PCL allocator ABI mismatch |
| A-002 | upstream arrival-order assumption exposed by ROS 2 execution | fixed, full Outdoor01 regression passed | pre-LiDAR IMU acceptance depended on callback scheduling rather than timestamps |
| A-003 | ROS 2 port defect | fixed, full Outdoor01 LIO regression passed | TF broadcaster was reconstructed on every odometry publication |
| A-004 | ROS 2 shutdown race | fixed, full Outdoor01 regression passed | shutdown could invalidate the ROS context between loop check, `spin_some`, and publication |
| A-005 | ROS 2 port defect | fixed, full Outdoor01 LIO regression passed; LIVO pending | `main.cpp` constructed an `ImageTransport` from a null node pointer |
| A-006 | upstream output-time contract defect | fixed, short regression passed | state-derived ROS messages used publication wall time instead of the state measurement time |
| A-007 | ROS 2 port defect | fixed, zero-offset M3DGR regression passed | ROS 2 dropped the official IMU/LiDAR time-offset parameter contract |
| A-008 | upstream output covariance defect | fixed, unit and short runtime regression passed | odometry published an all-zero covariance despite a nonzero internal ESIKF covariance |
| A-009 | upstream output completeness limitation | under investigation | odometry twist and twist covariance remain default zero although velocity is estimated |

## A-001: Eigen/PCL allocator ABI mismatch

### Source and mathematical/ABI explanation

- ROS 2 baseline: `CMakeLists.txt` added `-march=native` to every x86 Release translation unit.
- HKU reference: the same optimization flag is present, so this is not uniquely introduced by the ROS 2 port.
- On this AVX2 host, compiling Eigen without `-march=native` gives `EIGEN_DEFAULT_ALIGN_BYTES=16` and `EIGEN_MALLOC_ALREADY_ALIGNED=1`. With `-march=native`, those values become 32 and 0.
- The system PCL shared library allocates a `PointCloud<PointXYZINormal>` vector buffer using its distro ABI. Inline destruction in the project binary used the wider Eigen alignment ABI and attempted to read allocator metadata eight bytes before the PCL allocation.

### Trigger and measured impact

1. Run LIO long enough to execute `pcl::VoxelGrid::filter`.
2. Stop the node with SIGINT after processing has drained.
3. All three Phase 1 full runs exited with code `-11`.
4. GDB reproduced the fault after a ten-second M3DGR prefix. The stack ended in `free -> shared_ptr<PointCloud<PointXYZINormal>>::_M_dispose -> LIVMapper::~LIVMapper`.
5. ASan with a drained 0.1x prefix reported a heap-buffer-overflow in `Eigen::internal::handmade_aligned_free`. The buffer was allocated by `libpcl_filters.so.1.12`; the invalid read was eight bytes before the allocation.

### Fix

Remove `-march=native` from the x86 Release flags while retaining `-O3`, `-mtune=native`, and loop unrolling. `-mtune` changes scheduling but does not enable a wider instruction/alignment ABI.

### Verification to date

- Clean ROS 2 Humble Release build: pass.
- `compile_commands.json` contains no `-march=native`: pass.
- Low-rate M3DGR prefix, drain, SIGINT: mapping process finished cleanly instead of `-11`.
- Full Outdoor01 three-run regression: pass; see the consolidated results below.

An ordinary ASan build is not ABI-compatible with Ubuntu's prebuilt PCL for this check: Eigen deliberately changes `EIGEN_MALLOC_ALREADY_ALIGNED` from 1 to 0 when `__SANITIZE_ADDRESS__` is defined, while the distro PCL binary retains its normal 16-byte malloc contract. That instrumentation mismatch independently reproduces `handmade_aligned_free` even after removing `-march=native`, so it is not evidence that the Release fix failed. A validation-only ASan build with `EIGEN_MALLOC_ALREADY_ALIGNED=1`, matching the system PCL ABI, processed the same short M3DGR prefix and destroyed the mapper without an ASan finding. This macro is not added to the production CMake configuration.

## A-002: arrival-dependent initial IMU window

### Source comparison

Both HKU ROS 1 and chaoo0 ROS 2 return immediately from the IMU callback while `last_timestamp_lidar < 0`. The accepted IMU prefix therefore depends on which subscription callback happens to run first, not only on sensor timestamps.

### Evidence and impact

- Phase 1 Run 1 initialized with progress `3.3%, 30.0%, 96.7%`; Runs 2/3 used `3.3%, 33.3%, 100%`.
- The resulting initial gravity values differed, and Run 1 later differed from Runs 2/3 by 0.258 m pairwise position RMSE.
- A debugger run showed the first LiDAR callback followed by nine accepted IMUs for the first update; prior IMUs had already been discarded.

【推断】ROS 2 per-subscription delivery and `spin_some` timing selected a different initial sample prefix across runs.

### Timestamp evidence and policy

- In `Outdoor01_ros2`, 12 IMU storage records precede the first LiDAR storage record.
- The first IMU header time is `1735888008.064248085`. The first LiDAR header time is `1735888008.019938469`, although that LiDAR message was stored at `1735888008.120844603`, about 100.906 ms after its scan-start header time.
- Therefore, those early IMUs are not pre-scan measurements: they lie inside the first LiDAR scan interval and were discarded only because their callbacks ran before the LiDAR callback.
- The accepted startup set is now defined by corrected sensor time, `t_imu >= t_lidar,0`, rather than callback arrival. IMUs received before the first valid LiDAR callback are buffered and replayed in per-topic order once `t_lidar,0` is known. Older corrected samples are rejected.
- The startup buffer is bounded at 10,000 messages and drops the oldest sample with a warning if LiDAR never arrives; this prevents an unbounded failure mode.

### Short regression evidence

- Clean ROS 2 Humble Release build: pass.
- Three independent four-second normal-rate prefixes each initialized at progress `3.3%`, then `40.0%`, and produced the same gravity `[4.4705, 0.2924, -8.7273]`.
- Each run produced 35 poses. All three trajectory files are byte-identical with SHA-256 `cdc63fdeca5900d915371e9893d07b83b8c32bf4eb15b612a0bd7862a2cee2de`.
- A later instrumented short run observed two IMU callbacks before the first LiDAR callback; callback placement can vary, while the timestamp-selected initialization result remains the same.
- These short runs established deterministic startup for the prefix. Full-sequence results are reported below.

## A-003 and A-004: TF construction and shutdown race

- HKU ROS 1 uses one function-static `tf::TransformBroadcaster`.
- The ROS 2 port assigns `std::make_shared<tf2_ros::TransformBroadcaster>(node)` on every odometry publication, repeatedly creating a ROS publisher.
- Under ASan slowdown, SIGINT arrived during a frame and the code subsequently attempted to create a publisher or wait set after the ROS context was invalid, throwing `rclcpp::exceptions::RCLError` and aborting.

The fix moves TF broadcaster construction to initialization and makes the run loop stop safely if shutdown occurs between processing stages. These changes are kept separate from A-001.

The implementation now creates the ROS node explicitly in `main`, passes the valid node to `LIVMapper`, uses the caller's `ImageTransport`, and constructs the TF broadcaster once with the other publishers. ROS entities are declared after the node member so they are destroyed before it. The run loop checks the node context between stages and treats `RCLError` as a clean stop only when the context is already shutting down; unrelated ROS errors are rethrown.

### Short regression evidence

- Clean ROS 2 Humble Release build: pass.
- Normal-rate ten-second M3DGR prefix interrupted during processing: mapper finished cleanly in Release.
- The same processing interruption in the PCL-ABI-matched ASan build: mapper completed the in-flight frame and finished cleanly, with no ASan finding.
- Idle PCL-ABI-matched ASan node interrupted by a single launch-forwarded SIGINT: mapper finished cleanly, with no ASan finding.
- ASan execution was slower than the sensor stream and emitted synchronization warnings, so these runs validate memory/lifecycle behavior only; they are not runtime or estimator-accuracy evidence.

## Full Outdoor01 regression after A-001 through A-005

All three trials used the frozen Phase 1 normal-rate protocol, independent ROS domain IDs and output directories, and no concurrent build or data audit.

| Metric | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Output poses / LIO map updates | 4,114 | 4,114 | 4,114 |
| First output timestamp | 1735888008.319917 | same | same |
| Last output timestamp | 1735888419.619859 | same | same |
| Position ATE RMSE (m) | 0.274319 | 0.274319 | 0.274319 |
| Position RPE 1 s RMSE (m) | 0.032148 | 0.032148 | 0.032148 |
| Position RPE 5 s RMSE (m) | 0.072829 | 0.072829 | 0.072829 |
| Position RPE 10 s RMSE (m) | 0.113196 | 0.113196 | 0.113196 |
| Mean frame time (ms) | 19.486 | 19.470 | 19.362 |
| p99 frame time (ms) | 36.836 | 36.173 | 37.254 |
| Wall duration (s) | 418 | 418 | 418 |

- The three trajectory files are byte-identical with SHA-256 `e378283efe00f262881182782d9ce82a0e5abcb865b35ef398ce4f405f22999b`.
- Pairwise position RMSE and maximum error are exactly zero. The reported approximately `2.08e-8` rad quaternion comparison value is floating-point normalization noise on identical text files, not trajectory divergence.
- Position ATE and 1/5/10-second RPE sample coefficients of variation are all zero, passing the predeclared 1% repeatability gate on this host and sequence.
- The initial callback-dependent buffer contained 2, 3, and 3 IMUs respectively, while initialization progress and gravity were identical. This is direct evidence that callback placement varied but no longer selected the initialization sample set.
- Every run had two initial empty-point updates, zero IMU/LiDAR loopbacks, zero synchronization warnings, full end-time coverage, and clean mapper shutdown.
- The new timestamp policy completes initialization one LiDAR frame earlier than the old baseline and therefore produces one additional pose. Its ATE is 0.042% above the best old run and 2.17% below the other two old runs. These comparisons do not establish an accuracy improvement; they show no material position-accuracy regression while removing the measured nondeterminism.
- The machine-readable three-run summary is stored at `results/phase2_hku_ros2_audit/Outdoor01/three_run_summary.json` outside the source repository and has SHA-256 `20a01699c51d90484f1ed81208746b9aab851c67b637d0268f4a70d71ad320e0`.
- The available GT remains position-only. None of these results validates rotational accuracy or full SE(3) accuracy.

## A-006: output messages stamped in the wrong time domain

### Source comparison and measured impact

- Both the HKU ROS 1 source and the chaoo0 ROS 2 port stamp state-derived odometry, TF, path poses, registered point clouds, effect points, and the processed image with publication wall time.
- The state itself is propagated or updated to `LidarMeasures.last_lio_update_time`; the TUM output already uses that sensor time.
- During an `Outdoor01` replay without `/clock`, one captured odometry pose matched trajectory row 6 at sensor time `1735888008.818572`, but its ROS header was `1789100439.163598001`. The 53,212,430.345-second difference prevents correct offline association and time-consistent TF lookup.
- The published `nav_msgs/Path` also retained its construction-time header stamp while appending poses with later publication stamps.

### Fix

All state-derived products now use `sec2Stamp(LidarMeasures.last_lio_update_time)`. Odometry and its TF share the same measurement stamp; each path pose and the path header are updated together. IMU-propagated odometry remains stamped with its corresponding IMU message and is unchanged.

This is a metadata correction and does not alter the estimator state, covariance, measurement grouping, or map contents.

### Verification

- Clean ROS 2 Humble Release build: pass.
- The post-fix capture of the same pose had ROS time `1735888008.818572282`; its TUM row was `1735888008.818572`. The 0.282-microsecond displayed difference is only the TUM file's six-decimal formatting.
- The 35-pose pre-fix and post-fix short trajectories are byte-identical with SHA-256 `cdc63fdeca5900d915371e9893d07b83b8c32bf4eb15b612a0bd7862a2cee2de`.
- The mapper finished cleanly with no loopback or synchronization warning. This validates the LIO publication path; the visual publication path remains subject to later LIVO testing.

## A-007: dropped input time-offset parameters in the ROS 2 port

### Source comparison and impact

- The HKU reference declares and reads both `time_offset/imu_time_offset` and `time_offset/lidar_time_offset`. It subtracts the IMU offset before synchronization and adds the LiDAR offset in the standard `sensor_msgs/PointCloud2` callback.
- The ROS 2 port retained `imu_time_offset` in the correction expression but never declared or read the parameter, leaving it permanently at its member default of zero.
- The ROS 2 port removed the `lidar_time_offset` member and used the uncorrected header stamp in the standard point-cloud callback.
- This defect does not affect the current M3DGR Livox CustomMsg run because its configured IMU offset is zero and the official custom-message path does not apply `lidar_time_offset`. It does make nonzero IMU calibration ineffective and breaks parity for standard point-cloud datasets such as the official NTU-VIRAL configuration.

### Fix

Restore both ROS 2 parameters with zero defaults and apply `lidar_time_offset` to the standard point-cloud scan time exactly where the HKU reference does. The M3DGR YAML now states explicitly that its zero LiDAR offset is unused by Livox CustomMsg.

### Verification

- Clean ROS 2 Humble Release build: pass.
- Runtime parameter introspection reports both `time_offset.imu_time_offset` and `time_offset.lidar_time_offset` as declared doubles with the configured M3DGR value `0.0`.
- The 35-pose zero-offset M3DGR regression is byte-identical to the pre-A-007 trajectory; both have SHA-256 `cdc63fdeca5900d915371e9893d07b83b8c32bf4eb15b612a0bd7862a2cee2de`.
- The mapper finished cleanly with no loopback or synchronization warning.
- 【未知】A real standard-PointCloud2 dataset with a calibrated nonzero LiDAR offset has not yet been replayed, so that branch has source-parity and build evidence but not dataset-level validation.

## A-008: zero odometry pose covariance

### Source and coordinate contract

- `StatesGroup::operator+` applies rotation error on the right, `R_new = R Exp(delta_theta_body)`, while position error is added directly in the world frame. The state covariance pose order is therefore `[delta_theta_body, delta_p_world]`.
- ROS `PoseWithCovariance` uses row-major order `[x, y, z, rotation about fixed X, fixed Y, fixed Z]` according to [REP-103](https://github.com/openrobotics/reps/blob/main/_posts/rep-0103.md), and the odometry pose is expressed in `header.frame_id`.
- For small errors, `R Exp(delta_theta_body) = Exp(R delta_theta_body) R`; therefore the published fixed/world-axis rotation error is `delta_theta_world = R delta_theta_body`.
- The 6D mapping uses the reordered covariance and `J = diag(I, R)`, including the position-rotation cross terms. Only the published 6x6 copy is symmetrized to remove round-off asymmetry; the filter covariance is not modified.

### Verification

- Added three unit tests covering identity-frame reordering, a 90-degree body-to-world tangent rotation, and cross-covariance transformation/symmetry. All passed.
- A runtime odometry sample contained 36 finite, nonzero covariance entries and was exactly symmetric. Its diagonal was approximately `[1.209e-5, 5.931e-6, 4.456e-6, 3.817e-7, 5.818e-7, 8.987e-7]` in `[m^2, rad^2]` order.
- The six eigenvalues were positive for that sample; the smallest was approximately `1.62e-7`.
- The 35-pose trajectory remained byte-identical to the pre-A-008 run with SHA-256 `cdc63fdeca5900d915371e9893d07b83b8c32bf4eb15b612a0bd7862a2cee2de`, and the mapper finished cleanly.
- 【未知】Positive semidefiniteness of one sample does not establish covariance consistency. NEES/NIS or empirical coverage against reliable full-pose GT remains required before using this covariance as a calibrated confidence signal.
- Twist fields are deliberately unchanged in this fix. Their frame and covariance require a separate contract (A-009).

## Next audit actions

1. Audit odometry twist semantics separately; do not infer angular velocity or its covariance from an unstated time/frame contract.
2. Audit buffer loopback/reset handling and all shared deque invariants.
3. Continue through IMU propagation and de-skew, LiDAR update/Jacobians, covariance, and voxel-map feedback.
4. Audit and test the visual path only after the LIO control flow and mathematical contracts are stable.
