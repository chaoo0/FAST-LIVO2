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
| A-001 | upstream build portability defect | fixed, short regression passed, full regression pending | x86 `-march=native` creates an Eigen/PCL allocator ABI mismatch |
| A-002 | upstream arrival-order assumption exposed by ROS 2 execution | confirmed, unfixed | pre-LiDAR IMU acceptance depends on callback scheduling rather than timestamps |
| A-003 | ROS 2 port defect | confirmed, unfixed | TF broadcaster is reconstructed on every odometry publication |
| A-004 | ROS 2 shutdown race | confirmed, unfixed | shutdown can invalidate the ROS context between loop check, `spin_some`, and publication |
| A-005 | ROS 2 port defect candidate | under investigation | `main.cpp` constructs an unused `ImageTransport` from a null node pointer |

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
- Full Outdoor01 three-run regression: pending.

The ASan post-fix run did not reproduce A-001, but it exposed A-004 before normal destruction, so it is not counted as an independent clean-shutdown pass.

## A-002: arrival-dependent initial IMU window

### Source comparison

Both HKU ROS 1 and chaoo0 ROS 2 return immediately from the IMU callback while `last_timestamp_lidar < 0`. The accepted IMU prefix therefore depends on which subscription callback happens to run first, not only on sensor timestamps.

### Evidence and impact

- Phase 1 Run 1 initialized with progress `3.3%, 30.0%, 96.7%`; Runs 2/3 used `3.3%, 33.3%, 100%`.
- The resulting initial gravity values differed, and Run 1 later differed from Runs 2/3 by 0.258 m pairwise position RMSE.
- A debugger run showed the first LiDAR callback followed by nine accepted IMUs for the first update; prior IMUs had already been discarded.

【推断】ROS 2 per-subscription delivery and `spin_some` timing select a different initial sample prefix across runs. A deterministic timestamp policy must be specified before changing this behavior; simply retaining every pre-LiDAR IMU could change the initialization interval and is not yet accepted as the fix.

## A-003 and A-004: TF construction and shutdown race

- HKU ROS 1 uses one function-static `tf::TransformBroadcaster`.
- The ROS 2 port assigns `std::make_shared<tf2_ros::TransformBroadcaster>(node)` on every odometry publication, repeatedly creating a ROS publisher.
- Under ASan slowdown, SIGINT arrived during a frame and the code subsequently attempted to create a publisher or wait set after the ROS context was invalid, throwing `rclcpp::exceptions::RCLError` and aborting.

The fixes must move TF broadcaster construction to initialization and make the run loop stop safely if shutdown occurs between processing stages. These changes will be committed separately from A-001.

## Next audit actions

1. Complete and commit A-001 as an isolated build fix.
2. Fix A-003/A-004 and reproduce both idle and mid-frame shutdown.
3. Define and test a timestamp-based first-IMU policy for A-002.
4. Re-run full Outdoor01 three times before accepting any repeatability claim.
5. Continue through IMU propagation, LiDAR update, covariance, voxel-map feedback, publication timestamps, and the visual path only after the input/lifecycle layer is stable.
