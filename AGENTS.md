# AGENTS.md

## Project Scope

This repository is currently being used for the **FAST-LIVO2 + MambaPose** research prototype.

The current main line of work is:

> **Innovation Point 1: Mamba-based temporal pose compensation inside FAST-LIVO2.**

Do not restart the project design from scratch. Do not expand into the later innovation points unless explicitly requested.

Before starting any task in this repository, read:

```text
mamba_pose_project_context.md
```

That file contains the current project state, verified results, key files, build commands, testing commands, and current next task.

---

## Current Main Objective

The current objective is not to redesign FAST-LIVO2. The objective is to continue from the existing MambaPose engineering prototype:

```text
FAST-LIVO2 LIO state estimation
        ↓
PoseCompensator
        ↓
history sequence features
        ↓
ONNX Runtime backend
        ↓
6D pose correction
        ↓
safety layer
        ↓
pose compensation path
```

The ONNX Runtime engineering loop has already been verified with normal / clamp / reject tests, and the same capability has now been synced to the MID360 launch flow.

The current offline data-preparation stage is:

> Keep the FAST-LIVO2 C++ main flow stable, use the existing offline scripts to clean exported CSV, interpolate short timestamp gaps, build fixed-length sequences, compute feature normalization statistics, build a static zero-label baseline dataset, and train / export a first no-op baseline ONNX model.

After every code-development or script-development task, always sync the relevant project Markdown documents before ending the task.

---

## Repository / Environment

- Repository: `chaoo0/FAST-LIVO2`
- Local package path: `/home/liu/fast_livo2/src/FAST-LIVO2`
- Workspace root: `/home/liu/fast_livo2`
- ROS version: ROS2 Humble
- ROS package name: `fast_livo`
- Build system: `colcon`
- Current main launch file: `launch/mapping_mid360.launch.py`
- ONNX Runtime root: `/opt/onnxruntime`

Current MID360 launch arguments:

```text
use_rviz
mamba_pose_params_file
play_bag
bag_path
bag_loop
bag_clock
```

Default rosbag path:

```text
/home/liu/rosbags/mid360_fastlivo_mamba_20260519_211645
```

Default build command:

```bash
cd /home/liu/fast_livo2
source /opt/ros/humble/setup.bash

colcon build --symlink-install --packages-select fast_livo \
  --cmake-args \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DENABLE_ONNXRUNTIME=ON \
  -DONNXRUNTIME_ROOT=/opt/onnxruntime

source /home/liu/fast_livo2/install/setup.bash
```

---

## Do Not Modify Without Explicit Request

Do not modify these files unless the user explicitly asks and the reason is clear:

```text
src/voxel_map.cpp
include/voxel_map.h
src/vio.cpp
include/vio.h
```

Also avoid modifying these unless the current task truly requires it:

```text
src/pose_compensator.cpp
include/pose_compensator.h
src/LIVMapper.cpp
include/LIVMapper.h
CMakeLists.txt
launch/mapping_avia.launch.py
```

The ONNX Runtime backend, safety layer, and PoseCompensator inference path have already passed tests. Do not refactor them unnecessarily.

---

## Current Expected Behavior

The following are already verified:

- MambaPose parameters load through ROS2 YAML.
- `PoseCompensator` enters the `handleLIO()` path.
- ONNX Runtime C++ is found by CMake.
- `fastlivo_mapping` links to `libonnxruntime.so.1.18.1`.
- ONNX backend runs successfully.
- normal / clamp / reject safety tests pass on the validated launch flow.
- The compensation chain is verified through:
  `handleLIO() -> StateEstimation() -> PoseCompensator -> history sequence -> ONNX Runtime backend -> 6D correction -> safety layer -> derived data rebuild -> UpdateVoxelMap()`
- training data CSV export works.
- `scripts/analyze_mamba_pose_train_data.py` is already implemented.
- `scripts/interpolate_mamba_pose_time_gaps.py` is already implemented.
- `scripts/build_mamba_pose_sequences.py` is already implemented.
- `scripts/compute_mamba_pose_feature_norm.py` is already implemented.
- `scripts/build_mamba_pose_static_zero_label_dataset.py` is already implemented.
- `scripts/train_mamba_pose_static_zero.py` is already implemented.
- `scripts/export_static_zero_mamba_pose_raw_input_onnx.py` is already implemented.

Current generated CSV:

```text
/home/liu/fast_livo2/src/FAST-LIVO2/Log/mamba_pose_train_data.csv
```

Formal export configuration:

```text
config/mamba_pose_train_export_dummy.yaml
```

Use that file for real training-data export so the FAST-LIVO2 state history is not polluted by ONNX test corrections.

Test-only YAML files:

```text
config/mamba_pose_onnx_normal_test.yaml
config/mamba_pose_onnx_safety_test.yaml
config/mamba_pose_onnx_reject_test.yaml
```

These are only for ONNX / safety validation and should not be used for formal training-data export.

Runtime observe-only YAML:

```text
config/mamba_pose_static_zero_runtime_observe_only.yaml
```

Use this file when validating the raw-input `static_zero` ONNX model inside FAST-LIVO2 without writing the predicted correction back into `_state`.

Current next work:

```text
Run runtime no-op verification with Log/models/static_zero_mamba_pose_raw_input.onnx in observe-only mode first, then later replace the label side with a real correction target on top of the same X-side pipeline.
```

The current offline artifacts now include a `static_zero` baseline label dataset, but that label is only valid for the current stationary rosbag and does not define the final real correction target y.
The raw-input ONNX wrapper is required because FAST-LIVO2 currently sends raw basic18 features to the ONNX backend, not pre-normalized features.
The runtime path now separates:

- `mamba_pose/enabled`: enable history buffering, ready-state checks, ONNX inference, and MambaPose debug logs
- `mamba_pose/apply_correction_en`: control whether the compensated state is actually written back into `_state` and `voxelmap_manager->state_`

Default for `mamba_pose/apply_correction_en` is `false`, so new runtime validation should start in observe-only mode.

---

## If Modifying Code

If any code is modified, always output:

1. Files changed.
2. Why each file was changed.
3. Whether core SLAM files were touched.
4. Build command.
5. Runtime verification command.
6. Expected logs or outputs.

For C++ changes, always compile with:

```bash
cd /home/liu/fast_livo2
source /opt/ros/humble/setup.bash

colcon build --symlink-install --packages-select fast_livo \
  --cmake-args \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DENABLE_ONNXRUNTIME=ON \
  -DONNXRUNTIME_ROOT=/opt/onnxruntime
```

For the current documentation-sync and offline data-preparation tasks, no C++ compilation should be needed.
