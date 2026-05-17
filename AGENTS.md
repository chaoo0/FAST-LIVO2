# AGENTS.md

## Project Scope

This repository is currently being used for the **FAST-LIVO2 + MambaPose** research prototype.

The current main line of work is:

> **Innovation Point 1: Mamba-based temporal pose compensation inside FAST-LIVO2.**

Do not restart the project design from scratch. Do not expand into the later innovation points unless explicitly requested.

Before starting any task in this repository, read:

```text
Supplementary/mamba_pose_project_context.md
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

The ONNX Runtime engineering loop has already been verified with normal / clamp / reject tests.

The current task is:

> Analyze and clean the exported training data CSV so that it can later be used to train a real Mamba Pose Correction model.

---

## Repository / Environment

- Repository: `chaoo0/FAST-LIVO2`
- Local package path: `/home/liu/fast_livo2/src/FAST-LIVO2`
- Workspace root: `/home/liu/fast_livo2`
- ROS version: ROS2 Humble
- ROS package name: `fast_livo`
- Build system: `colcon`
- Current launch file: `launch/mapping_avia.launch.py`
- ONNX Runtime root: `/opt/onnxruntime`

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
- normal / clamp / reject safety tests pass.
- training data CSV export works.

Current generated CSV:

```text
/home/liu/fast_livo2/src/FAST-LIVO2/Log/mamba_pose_train_data.csv
```

Current next work:

```text
scripts/analyze_mamba_pose_train_data.py
```

This script should analyze and clean the CSV data. It should not modify the FAST-LIVO2 C++ main flow.

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

For the current data-analysis task, no C++ compilation should be needed.
