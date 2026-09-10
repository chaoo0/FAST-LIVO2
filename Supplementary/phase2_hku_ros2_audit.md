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
| A-003 | ROS 2 port defect | fixed, short regression passed, full regression pending | TF broadcaster was reconstructed on every odometry publication |
| A-004 | ROS 2 shutdown race | fixed, short regression passed, full regression pending | shutdown could invalidate the ROS context between loop check, `spin_some`, and publication |
| A-005 | ROS 2 port defect | fixed, short regression passed, full regression pending | `main.cpp` constructed an `ImageTransport` from a null node pointer |

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

An ordinary ASan build is not ABI-compatible with Ubuntu's prebuilt PCL for this check: Eigen deliberately changes `EIGEN_MALLOC_ALREADY_ALIGNED` from 1 to 0 when `__SANITIZE_ADDRESS__` is defined, while the distro PCL binary retains its normal 16-byte malloc contract. That instrumentation mismatch independently reproduces `handmade_aligned_free` even after removing `-march=native`, so it is not evidence that the Release fix failed. A validation-only ASan build with `EIGEN_MALLOC_ALREADY_ALIGNED=1`, matching the system PCL ABI, processed the same short M3DGR prefix and destroyed the mapper without an ASan finding. This macro is not added to the production CMake configuration.

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

The fix moves TF broadcaster construction to initialization and makes the run loop stop safely if shutdown occurs between processing stages. These changes are kept separate from A-001.

The implementation now creates the ROS node explicitly in `main`, passes the valid node to `LIVMapper`, uses the caller's `ImageTransport`, and constructs the TF broadcaster once with the other publishers. ROS entities are declared after the node member so they are destroyed before it. The run loop checks the node context between stages and treats `RCLError` as a clean stop only when the context is already shutting down; unrelated ROS errors are rethrown.

### Short regression evidence

- Clean ROS 2 Humble Release build: pass.
- Normal-rate ten-second M3DGR prefix interrupted during processing: mapper finished cleanly in Release.
- The same processing interruption in the PCL-ABI-matched ASan build: mapper completed the in-flight frame and finished cleanly, with no ASan finding.
- Idle PCL-ABI-matched ASan node interrupted by a single launch-forwarded SIGINT: mapper finished cleanly, with no ASan finding.
- ASan execution was slower than the sensor stream and emitted synchronization warnings, so these runs validate memory/lifecycle behavior only; they are not runtime or estimator-accuracy evidence.

## Next audit actions

1. Commit A-003/A-004/A-005 separately from the completed A-001 build fix.
2. Define and test a timestamp-based first-IMU policy for A-002.
3. Re-run full Outdoor01 three times before accepting any repeatability claim.
4. Continue through IMU propagation, LiDAR update, covariance, voxel-map feedback, publication timestamps, and the visual path only after the input/lifecycle layer is stable.
