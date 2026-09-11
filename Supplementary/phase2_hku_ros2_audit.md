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
| A-009 | upstream output completeness limitation | confirmed; no unsafe partial fix applied | odometry twist and twist covariance remain default zero although velocity is estimated |
| A-010 | ROS 2 port defect | fixed, ownership tests passed; LIVO replay pending | queued images could outlive the ROS message storage shared by their `cv::Mat` |
| A-011 | ROS 2 port defect | fixed, unit/synthetic/normal regression passed | one IMU gap over 0.2 s caused every later IMU message to be rejected |
| A-012 | upstream input-buffer defect | fixed, synthetic and normal regressions passed | LiDAR time reversal cleared only clouds, not their paired timestamps; unusable scans could stall the queue |
| A-013 | ROS 2 port defect | fixed, unit/runtime/normal regression passed | source read `lio.min_iterations` while upstream and shipped configurations set `lio.max_iterations` |
| A-014 | ROS 2 port defect | fixed, synthetic/normal regression passed | Ouster point-time sorting was removed and Pandar128 absolute time/schema replaced the required relative scan-time contract |
| A-015 | upstream point-covariance frame defect | fixed, mathematical/unit/full Outdoor01 regression passed; cross-sequence validation pending | map-point pose uncertainty omitted the world rotation and rotation-position cross covariance |
| A-016 | audit-tool numerical reporting defect | fixed, unit-tested | pairwise comparison amplified one-ulp quaternion normalization noise into a fictitious nonzero angle |
| A-017 | upstream zero-effective-feature robustness defect | confirmed by control-flow inspection; dataset trigger and fix pending | zero matched planes are divided into the residual average and passed into a zero-row ESIKF update |

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

## A-009: odometry twist is not a valid velocity estimate

### Contract and source evidence

- The installed ROS 2 `nav_msgs/msg/Odometry` definition requires pose in `header.frame_id` and twist in `child_frame_id`.
- Both the frozen HKU source and ROS 2 port leave `/aft_mapped_to_init.twist` and its covariance at their default zeros, even though `_state.vel_end` is estimated and explicitly documented as world-frame velocity.
- The optional `/LIVO2/imu_propagate` path writes that world-frame velocity into `twist.linear`, sets `header.frame_id = world`, and leaves `child_frame_id` empty. That is not the required child-frame twist contract.

### Why no partial patch is accepted

- A body-frame linear velocity can be computed as `R_world_body^T * v_world`, but its covariance is not merely a reordered state sub-block. Its first-order Jacobian includes both velocity and right/body rotation error: `delta_v_body = skew(v_body) * delta_theta_body + R_world_body^T * delta_v_world`.
- The filter does not carry angular velocity as a state. A correct angular component requires a measurement-time raw gyroscope sample minus the posterior bias, plus a declared noise/cross-covariance model. The cached propagation angular rate was bias-corrected using the prior state and is not automatically posterior-consistent after the LiDAR update.
- Filling only selected fields while leaving zero angular velocity/covariance would continue to assert false certainty and could break downstream consumers. Therefore A-009 is recorded as a confirmed output limitation rather than disguised as a complete fix.

【影响】This does not change the internal ESIKF pose trajectory or map, but `/aft_mapped_to_init` must not be used as a trusted six-degree-of-freedom twist source. A future publication-interface change requires an explicit frame/noise contract and consumer regression; Mamba diagnostics must read the audited estimator state directly rather than infer motion from this zero twist.

## A-010: queued image lifetime lost in the ROS 2 port

### Source comparison and failure mechanism

- The frozen HKU implementation converts each ROS image with `cv_bridge::toCvCopy(..., "bgr8")` before storing the resulting `cv::Mat`.
- The ROS 2 migration changed this to `toCvShare()` while `img_buffer` continued to store only `cv::Mat`; it does not retain the returned `CvImageConstPtr` or the ROS message.
- When the source encoding is already `bgr8`, `toCvShare()` points the matrix at the ROS message's external byte buffer. Copying only the matrix header does not retain the tracked ROS-message owner. Once the callback-local message and temporary `CvImageConstPtr` are destroyed, the queued matrix can dangle.
- This path is disabled in the accepted Phase-1 LIO runs, so the defect cannot explain their trajectory. It can cause invalid image reads or nondeterministic visual residuals when LIVO is enabled.

### Fix and verification

- Restore `toCvCopy()` so the queued `cv::Mat` owns a reference-counted OpenCV allocation independent of the ROS message lifetime.
- A structural ownership test using a same-encoding `bgr8` message confirmed that the `toCvShare()` matrix aliases the message buffer and does not keep the message alive.
- A regression test confirmed that the `toCvCopy()` matrix uses independent storage and preserves both test pixels after the ROS message expires.
- Clean Release build and the complete current unit-test set passed: 7 tests, 0 errors, 0 failures, 0 skipped.
- 【未知】No dataset-level LIVO replay is claimed here. Camera calibration and the published 0.1 s M3DGR image offset must be validated before a LIVO trajectory can serve as regression evidence.

## A-011: permanent IMU rejection after one forward gap

### Source comparison and failure mechanism

- The frozen HKU callback contains a disabled/commented check for an IMU interval greater than 0.2 s. The ROS 2 migration enabled that branch and returns before updating `last_timestamp_imu`.
- If the stored time is `t` and the next sample is `t + 0.3`, that sample is rejected and the stored time remains `t`. Every later monotonically increasing sample is also greater than `t + 0.2`, so the input remains permanently frozen without an explicit restart.
- A forward gap indicates missing propagation data and must be reported, but it is not a time reversal. Silently freezing all later samples is neither recovery nor fail-closed state reinitialization.

### Fix and verification

- Classify first/in-order, forward-gap, backward, and invalid timestamps explicitly. Backward and non-finite IMU stamps remain rejected; a forward gap is warned and accepted so time can advance.
- Three classifier tests cover the 0.2 s boundary, forward/backward distinction, and non-finite inputs.
- A ROS 2 callback-level test sent a real Outdoor01 MID360 point cloud followed by IMU stamps at `8.064`, `8.364`, and `8.369` s within the same epoch. The 0.3 s interval produced one warning and all three messages were accepted.
- Outdoor01 contains 82,312 IMU records with zero header inversions and an observed maximum gap of about 6.71 ms, so the 0.2 s branch is not exercised by the accepted baseline dataset.
- A normal short replay produced 31 poses byte-identical to the first 31 poses of the A-008 reference. Both prefixes have SHA-256 `d95c51ba6ad9a8e7574ef540013cbda368b5b1a2f5ae06d9ee6963832286070d`; the mapper exited cleanly with no time-jump or loopback warning.
- 【未知】Accepting data after a large gap prevents the permanent software freeze but does not make the propagated state accurate across missing IMU coverage. Gap duration must become a diagnostic/failure input, and accuracy after such gaps requires a separate controlled experiment.

## A-012: LiDAR queue desynchronization and unusable-scan stall

### Source comparison and failure mechanism

- Both the frozen HKU source and the ROS 2 port clear `lid_raw_data_buffer` when LiDAR time moves backward but leave `lid_header_time_buffer` unchanged. The producer then appends one new cloud and one new timestamp, while `sync_packages()` assumes both deques have equal length and pops them in lockstep.
- Once the old timestamp prefix is longer than the repeated input prefix, a new cloud can be paired with an unrelated old time. That corrupts scan begin/end time, IMU selection, de-skew, and the state/map update chronology.
- The standard callback had no post-preprocessing empty check; the Livox callback rejected only exactly zero points. In `ONLY_LIO`, a scan with at most one usable point makes `sync_packages()` return before popping it, causing permanent head-of-line blocking.
- Clearing both queues and continuing is still insufficient after a real epoch reset because the ESIKF state, IMU integrator, voxel map, VIO state, and published trajectory are not atomically reset by this code.

### Fix and verification

- Reject non-finite and non-increasing LiDAR timestamps before preprocessing/enqueue, preserving all existing data/time pairs. The diagnostic explicitly requires a node restart for a new time epoch.
- Reject scans with fewer than two usable points in both LiDAR callbacks before they reach the synchronization queue.
- Check the LiDAR-data/time and image-data/time deque size invariants at the `sync_packages()` boundary. Any internal mismatch is fatal rather than allowing a wrong timestamp association.
- Replaying an initial LiDAR-only segment twice caused the repeated older frames to be rejected without an invariant failure. That experiment also exposed an equal-timestamp duplicate, after which the guard was tightened from decreasing to non-increasing.
- A final callback-level test accepted one two-point Livox scan at 100 s, rejected a second scan at the same stamp before preprocessing, then rejected a zero-point scan at 101 s before enqueue. The node exited cleanly.
- A normal Outdoor01 short replay produced 35 poses byte-identical to A-008, with SHA-256 `cdc63fdeca5900d915371e9893d07b83b8c32bf4eb15b612a0bd7862a2cee2de`; there were no rejection, time-jump, or queue-invariant diagnostics.
- 【未知】The standard `PointCloud2` callback shares the same source-level guards but has not yet received a dataset-level nonempty/empty regression. Cross-bag in-process reset remains deliberately unsupported.

## A-013: LIO maximum-iteration parameter typo

### Source comparison and impact

- The frozen HKU implementation reads `lio/max_iterations` into `VoxelMapConfig::max_iterations_`, and every shipped ROS 2 dataset configuration also uses `lio.max_iterations`.
- The ROS 2 loader instead declared/read `lio.min_iterations`. Consequently, changing the documented `max_iterations` key had no effect and the estimator silently retained the default value 5.
- The Phase-1 M3DGR configuration temporarily set both names to 5 only to preserve baseline behavior. That workaround did not make the source contract correct.

### Fix and verification

- Restore `lio.max_iterations` as the canonical parameter. If and only if an existing configuration supplies the old `min_iterations` typo without the canonical key, accept it with a deprecation warning; if both are present, the canonical key wins.
- Reject non-positive selected values at startup instead of running an empty/invalid iterative update.
- Four unit tests cover canonical selection, legacy-only compatibility, canonical precedence, and invalid values.
- Runtime parameter inspection with the M3DGR configuration reports `lio.max_iterations = 5`; `lio.min_iterations` is absent. The temporary duplicate key was removed from the M3DGR YAML.
- A normal Outdoor01 short replay produced 35 poses byte-identical to A-008/A-012, with SHA-256 `cdc63fdeca5900d915371e9893d07b83b8c32bf4eb15b612a0bd7862a2cee2de`, and the mapper exited cleanly.
- 【未知】Iteration counts other than 5 are now wired correctly but have not been claimed to improve accuracy or convergence. They require parameter-sweep evidence and must not be tuned on the frozen test bags.

## A-014: broken Ouster/Pandar128 per-point time contract

### Source comparison and failure mechanism

- The frozen HKU Ouster path sorts output points by their per-point time. The ROS 2 port removed that sort. `sync_packages()` uses the last point's offset as scan end time, and `UndistortPcl()` traverses points backward assuming nondecreasing offsets; unordered input therefore selects the wrong time bound and applies motion compensation in the wrong temporal order.
- The frozen HKU Pandar128 point schema is `x/y/z float32, intensity uint8, timestamp float64, ring uint16`. The ROS 2 port changed it to `timestamp float32, ring uint8`, omitted intensity from both the struct and PCL registration, and then computed `curvature = timestamp * 1000` rather than `(timestamp - first_timestamp) * 1000` milliseconds.
- With a synthetic scan whose point timestamps are `100.00, 100.02, 100.01` s, the broken formula produces offsets near 100,000 ms, so a header at 100 s is interpreted as ending near 200 s instead of 100.02 s. Synchronization can then wait for IMU data far outside the scan and de-skew uses an invalid duration.
- M3DGR MID360 uses the Livox CustomMsg handler, so neither broken branch is exercised by the accepted MID360 regression. This is a ROS 2 portability defect, not an explanation for the existing M3DGR trajectory.

### Fix and verification

- Restore Ouster time sorting.
- Restore the upstream Pandar128 PointField types, intensity mapping, relative scan time, and sorting; accept an empty Pandar cloud without indexing point zero.
- Three PointCloud2-level tests cover unordered Ouster offsets, Pandar field/time/intensity conversion, and an empty Pandar cloud. The complete suite passes: 20 tests, 0 errors, 0 failures, 0 skipped.
- Clean ROS 2 Humble Release build passes.
- A normal-rate Outdoor01 MID360 prefix produced 61 poses and exited cleanly. Its first 35 poses are byte-identical to the pre-A-014 A-013 reference, with SHA-256 `cdc63fdeca5900d915371e9893d07b83b8c32bf4eb15b612a0bd7862a2cee2de`.
- 【未知】No real Ouster or Pandar128 bag has been replayed. The repair has official-source parity and synthetic PointCloud2 evidence, not dataset-level accuracy evidence for those sensors.

## A-015: inconsistent world-frame map-point covariance

### Source locations and classification

- The defect is present in both the frozen HKU source and the ROS 2 port in `VoxelMapManager::StateEstimation()` and `VoxelMapManager::BuildVoxelMap()`. The same expression is repeated before `UpdateVoxelMap()` in `LIVMapper::handleLIO()`, so this is classified as an upstream algorithm defect rather than a ROS 2 migration defect.
- The initial-map path also formed the rotation Jacobian from the raw LiDAR point, whereas the estimated pose rotates the IMU-frame point `p_i = R_il p_l + t_il`. It therefore ignored both the LiDAR-to-IMU rotation and translation in that part of the uncertainty propagation.

### Mathematical explanation and trigger

- `StatesGroup::operator+=` defines a right rotation perturbation and a world-frame position perturbation: `R_new = R Exp(delta_theta_body)` and `t_new = t + delta_t_world`.
- For `p_w = R_wi p_i + t_wi`, first-order perturbation gives `delta p_w = -R_wi [p_i]_x delta_theta_body + delta t_world`. The full pose Jacobian is therefore `J_pose = [-R_wi [p_i]_x, I]`.
- The old implementation used `-[p_i]_x P_rr (-[p_i]_x)^T + P_tt` and added that result to a world-frame LiDAR point covariance. It omitted the leading `R_wi`, omitted `P_rt/P_tr`, and in the initial-map path used the wrong point in the skew matrix. The result is not equivariant to a change of world basis.
- The defect is triggered whenever the attitude is non-identity, the pose covariance has rotation-position cross terms, or the LiDAR-to-IMU extrinsic is nontrivial. It changes point-to-plane association uncertainty and voxel-plane uncertainty; those quantities affect residual acceptance and are then written into the persistent map, so the impact can feed forward into later estimates.

### Fix and verification

- Added one shared `worldPointCovariance()` implementation and replaced all three inconsistent propagation sites. It rotates the LiDAR point covariance with `R_wi R_il`, uses the full state covariance through `J_pose P J_pose^T`, and only symmetrizes the derived 3x3 point covariance to remove floating-point asymmetry.
- Three independent checks cover the analytic covariance expression, a central finite-difference Jacobian for the actual right perturbation, and covariance equivariance under an arbitrary world-frame rotation.
- ROS 2 Humble Release build passes. The complete suite passes: 24 tests, 0 errors, 0 failures, 0 skipped; this includes the finite-difference/equivariance covariance tests and the quaternion-comparison tool tests.
- A normal-rate 10 s Outdoor01 prefix produced 95 finite eight-field poses without a time-jump or queue-invariant diagnostic. Against the first 35 timestamp-identical pre-A-015 poses, the unoptimized diagnostic implementation changed position by 3.411 mm RMSE (5.907 mm maximum) and orientation by 1.166 mrad RMSE (3.509 mrad maximum).
- The production implementation was then reduced from a per-point 3x19 multiplication to the mathematically identical 3x6 pose block. The optimized full Outdoor01 replay produced 4,114 poses in all three runs, with identical SHA-256 `f42ecf88c660b5c66ec30a5cae9e07bcc851ac39c4bd8bdcf572da27f8aa08d0`. Mean frame times were 18.889, 18.957, and 18.891 ms; p99 times were 34.455, 35.489, and 34.572 ms; no run had a shutdown SIGSEGV, loopback, or out-of-sync diagnostic. One maximum frame reached 116.308 ms, so the mean/p99 real-time margin passes but the maximum-frame bound does not become a hard guarantee.
- Fixed-protocol position-only metrics were identical across the three optimized runs: ATE RMSE `0.271441 m`, RPE-1s `0.031956 m`, RPE-5s `0.072423 m`, and RPE-10s `0.112471 m`; sample CV was zero. Relative to the accepted pre-A-015 reference run (`0.274319/0.032148/0.072829/0.113196 m`), this is ATE -1.05%, RPE-1s -0.60%, RPE-5s -0.56%, and RPE-10s -0.64% for this one sequence. This is a small mixed-sequence result, not evidence of a general accuracy improvement or statistical significance.
- A-016 fixes the repeatability comparator so identical printed quaternions are exactly zero distance within floating-point round-off; this changes reporting only, not trajectories or estimator code.
- 【未知】The full result is still position-only because Outdoor01 GT orientations are identity. Dynamic/Varying-illu/Sha-turn full-pose GT audits are complete, but cross-sensor rigid-body alignment and resampled motion labels are not yet validated for a local 6D research target.

### First-pass regression on the six additional M3DGR sequences

The same optimized LIO binary, M3DGR MID360 configuration, position-only alignment protocol, and single normal-speed replay were used for each sequence. Varying-illu02 uses the frozen official GT copy because the local same-named file was previously shown to be a different sequence.

| Sequence | LiDAR input | Output poses | ATE RMSE (m) | RPE 1 s (m) | RPE 5 s (m) | RPE 10 s (m) | Mean / p99 frame (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dynamic01 | 1,752 | 1,749 | 0.141734 | 0.029921 | 0.111994 | 0.185504 | 24.056 / 31.603 |
| Dynamic02 | 1,502 | 1,499 | 0.137626 | 0.030827 | 0.113869 | 0.175315 | 23.367 / 32.299 |
| Varying-illu01 | 1,541 | 1,538 | 0.139514 | 0.033722 | 0.126565 | 0.201807 | 23.385 / 31.961 |
| Varying-illu02 | 1,465 | 1,462 | 0.138227 | 0.033093 | 0.127546 | 0.200847 | 23.851 / 32.297 |
| Sha-turn01 | 1,390 | 1,387 | 0.200092 | 0.063261 | 0.237857 | 0.348379 | 25.892 / 33.775 |
| Sha-turn02 | 1,004 | 1,002 | 0.144480 | 0.051734 | 0.173198 | 0.259183 | 26.162 / 33.297 |

- All six runs produced finite, timestamp-increasing trajectories with full GT timestamp matching and no shutdown SIGSEGV, IMU/LiDAR loopback, or out-of-sync diagnostic. The maximum individual frame times were 46.418–79.781 ms for these six runs.
- These metrics are position-only despite the valid VRPN quaternion fields because the current evaluator intentionally fixes the position protocol and does not yet compensate an independently verified rigid transform between the VRPN body and estimator IMU. They are baseline coverage, not a full-pose accuracy claim.
- The dynamic and turn sequences are not clean static-scene tests: moving objects and high angular motion can alter point-to-plane residual statistics and map consistency. Separating those effects requires the planned diagnostics, not post-hoc attribution from ATE alone.
- Artifacts are under `/home/liu/fast_livo2/results/phase2_hku_ros2_audit/A015_<sequence>/`; each directory contains `trajectory.tum`, `position_metrics.json`, `run_summary.txt`, `summary.json`, and ROS logs.

## A-017: zero-effective-feature path is not fail-closed

### Source comparison and trigger

- Both the frozen HKU source and the ROS 2 port compute `total_residual / effct_feat_num_` immediately after `BuildResidualListOMP()`, without checking whether `effct_feat_num_` is zero. They then construct `Hsub`, `Hsub_T_R_inv`, `R_inv`, and `meas_vec` with zero rows and continue into the ESIKF matrix inverse.
- A reproducible trigger is a scan whose transformed points find no valid voxel plane: an empty/isolated local map, a deliberately over-restrictive plane gate, or a synthetic teleport between two scans. The minimum experiment is to force that condition after map initialization and capture the residual log, state finiteness, covariance finiteness, and subsequent map update.

### Evidence and impact boundary

- This is a confirmed divide-by-zero/control-flow defect by inspection, classified as an upstream robustness defect. It is not yet a confirmed trajectory defect: the zero-row Eigen products and the subsequent `P.inverse()` behavior still require the synthetic trigger experiment.
- The six new M3DGR runs and the three Outdoor01 A015 runs had no `effective feature num: 0` record; their minimum effective feature counts were in the thousands. Therefore current GT metrics provide no evidence about this path.
- Until the synthetic trigger is run, no fix is committed. The safe candidate is an explicit fail-closed branch that retains the propagated state/covariance, reports the rejected update, and lets the caller perform only the separately audited map-write policy.

## Next audit actions

1. Extend the accepted A-015 LIO regression to the newly available Dynamic/Varying-illu/Sha-turn sequence matrix, using the audited GT files and recording frame/GT alignment limits per sequence.
2. Continue the LiDAR point-to-plane residual/Jacobian, covariance-update, convergence, and zero-effective-feature audit.
3. Add diagnostics only after their mathematical frame/time contracts are written; do not create a Mamba module.
4. Audit and test the visual path only after the LIO control flow and mathematical contracts are stable.
