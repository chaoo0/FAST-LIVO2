# FAST-LIVO2 + MambaPose Project Context

## 1. Project Identity

This repository is being used for a research prototype:

> **FAST-LIVO2 + MambaPose temporal pose compensation**

Repository:

```text
chaoo0/FAST-LIVO2
```

Local paths:

```text
Workspace root:
  /home/liu/fast_livo2

Package path:
  /home/liu/fast_livo2/src/FAST-LIVO2
```

Environment:

```text
ROS version:
  ROS2 Humble

Package name:
  fast_livo

Build tool:
  colcon

Main launch file currently used:
  /home/liu/fast_livo2/src/FAST-LIVO2/launch/mapping_avia.launch.py
```

Current ONNX Runtime path:

```text
/opt/onnxruntime
```

Current training CSV:

```text
/home/liu/fast_livo2/src/FAST-LIVO2/Log/mamba_pose_train_data.csv
```

---

## 2. Research Goal

The overall goal is to introduce **Mamba / temporal sequence modeling** into FAST-LIVO2.

FAST-LIVO2 is strong at local geometric optimization, but it does not explicitly model long-term temporal dependencies. This project builds a research prototype that lets a sequence model observe recent LIO state history and output a pose correction before map update.

Current intended pipeline:

```text
FAST-LIVO2 LIO StateEstimation()
        ↓
current state _state
        ↓
PoseCompensator
        ↓
history sequence construction
        ↓
ONNX Runtime model inference
        ↓
6D pose correction
        ↓
safety layer: finite / clamp / reject
        ↓
pose compensation path
        ↓
rebuild derived LIO data
        ↓
UpdateVoxelMap()
```

The current focus is **engineering validation and data preparation** for the first innovation point.

---

## 3. Three Planned Innovation Points

### 3.1 Innovation Point 1: Mamba Temporal Pose Compensation

This is the current main line.

Goal:

> Use recent LIO state history as sequence input and let a temporal model output a 6D pose correction.

Insertion point in `handleLIO()`:

```cpp
voxelmap_manager->StateEstimation(state_propagat);
_state = voxelmap_manager->state_;
_pv_list = voxelmap_manager->pv_list_;

// PoseCompensator is inserted here.

rebuildLioDerivedDataAfterCompensation();

// Then continue to map update.
```

Model output definition:

```text
[d_roll, d_pitch, d_yaw, d_tx, d_ty, d_tz]
```

Status:

```text
Engineering loop verified.
ONNX Runtime loop verified.
normal / clamp / reject tests passed.
Training CSV export verified.
```

### 3.2 Innovation Point 2: Mamba Point Cloud / Local Map Temporal Feature Enhancement

Not started yet.

Planned idea:

> Use historical point cloud, voxel, or local map context as temporal features to improve point weighting, residual construction, or local geometric stability.

Possible insertion point:

```text
after downsampling / transformLidar
before StateEstimation()
```

Do not start this unless explicitly asked.

### 3.3 Innovation Point 3: Mamba Implicit Loop Closure / Global Memory

Not started yet.

Planned idea:

> Use long-term temporal memory to assist implicit loop detection, global consistency, or revisit recognition.

Do not start this unless explicitly asked.

---

## 4. Current Stage

Current stage:

> Innovation Point 1 has passed the engineering ONNX Runtime validation. The training-data analysis and cleaning script has now been implemented and verified.

Already verified:

```text
PoseCompensator enters FAST-LIVO2 main flow.
history reaches ready state.
ONNX Runtime C++ backend is linked and active.
normal / clamp / reject tests pass.
training data CSV is generated.
training data analysis script runs successfully.
clean CSV export works.
TXT report export works.
```

Current next task:

```text
Use the clean CSV as the base dataset and continue with larger-scale data collection and training-oriented sequence preparation.
```

Current analysis inputs / outputs:

```text
/home/liu/fast_livo2/src/FAST-LIVO2/Log/mamba_pose_train_data.csv
/home/liu/fast_livo2/src/FAST-LIVO2/Log/mamba_pose_train_data_clean.csv
/home/liu/fast_livo2/src/FAST-LIVO2/Log/mamba_pose_train_data_report.txt
```

---

## 5. Completed Engineering Work

### 5.1 PoseCompensator Module

Added:

```text
include/pose_compensator.h
src/pose_compensator.cpp
```

Responsibilities:

- State history cache.
- Feature sequence construction.
- Flat input construction.
- Backend abstraction.
- Dummy backend.
- ONNX placeholder backend.
- Real ONNX backend.
- Pose correction decoding.
- Safety layer.
- Backend status and inference status reporting.

### 5.2 LIVMapper Integration

Modified:

```text
include/LIVMapper.h
src/LIVMapper.cpp
```

Added:

- `PoseCompensator` member.
- `applyPoseCompensationIfNeeded(...)`.
- `rebuildLioDerivedDataAfterCompensation()`.
- MambaPose parameter reading.
- MambaPose debug logs.
- Training data CSV export.

### 5.3 Rebuild Derived LIO Data After Compensation

Function:

```text
rebuildLioDerivedDataAfterCompensation()
```

Rebuilds:

```text
feats_down_world
voxelmap_manager->feats_down_world_
voxelmap_manager->pv_list_[i].point_w
voxelmap_manager->pv_list_[i].var
_pv_list
pcl_w_wait_pub
```

Does not rebuild:

```text
feats_down_body
feats_undistort
voxelmap_manager->cross_mat_list_
voxelmap_manager->body_cov_list_
voxelmap_manager->ptpl_list_
voxel_map_
```

### 5.4 Training Data Export

Parameters:

```text
mamba_pose/export_train_data_en
mamba_pose/export_train_data_path
```

Currently enabled in:

```text
config/mamba_pose_onnx_normal_test.yaml
```

Current values:

```yaml
"mamba_pose/export_train_data_en": true
"mamba_pose/export_train_data_path": "/home/liu/fast_livo2/src/FAST-LIVO2/Log/mamba_pose_train_data.csv"
```

CSV columns:

```text
timestamp,
pos_x,pos_y,pos_z,
rot_x,rot_y,rot_z,rot_w,
vel_x,vel_y,vel_z,
bias_g_x,bias_g_y,bias_g_z,
bias_a_x,bias_a_y,bias_a_z,
effective_feature_num,avg_residual,history_size,ready_flag,
gt_pos_x,gt_pos_y,gt_pos_z,
gt_rot_x,gt_rot_y,gt_rot_z,gt_rot_w
```

Ground-truth fields are currently `nan`, which is expected. They are reserved for future supervision.

### 5.5 Backend Abstraction

Supported backend types:

```text
dummy
onnx_placeholder
onnx
```

Parameter:

```text
mamba_pose/backend_type
```

Current normal test uses:

```text
onnx
```

### 5.6 Safety Layer

Safety parameters:

```text
mamba_pose/max_rotation_correction_rad
mamba_pose/max_translation_correction_m
mamba_pose/reject_non_finite_output
mamba_pose/reject_oversized_output
```

Safety behavior:

- Check correction dimension.
- Check NaN / Inf.
- Clamp oversized outputs.
- Optionally reject oversized outputs.
- Fall back to original state when rejected.

### 5.7 Training Data Analysis Script

Added:

```text
scripts/analyze_mamba_pose_train_data.py
```

Implemented capabilities:

- Default input CSV path handling.
- CLI arguments: `--input`, `--output-clean`, `--output-report`, `--save-clean`.
- Required-column validation.
- Basic statistics for history size, effective feature count, residual, pose, velocity, and bias fields.
- Abnormal-frame counting for invalid readiness, short history, invalid feature count, invalid residual, and NaN / Inf in input fields.
- Default valid-sample filtering.
- TXT report generation.
- Optional clean CSV export.
- Clear dependency hint for `pandas` and `numpy`.

Current verified outputs:

```text
Log/mamba_pose_train_data_report.txt
Log/mamba_pose_train_data_clean.csv
```

---

## 6. Key Files

### C++ / CMake

```text
include/pose_compensator.h
src/pose_compensator.cpp
include/LIVMapper.h
src/LIVMapper.cpp
CMakeLists.txt
```

### Launch

```text
launch/mapping_avia.launch.py
```

### Config

```text
config/mamba_pose_onnx_test.yaml
config/mamba_pose_onnx_normal_test.yaml
config/mamba_pose_onnx_safety_test.yaml
config/mamba_pose_onnx_reject_test.yaml
```

### Scripts

```text
scripts/export_dummy_mamba_pose_onnx.py
scripts/export_small_nonzero_mamba_pose_onnx.py
scripts/analyze_mamba_pose_train_data.py
```

### Documents

```text
Supplementary/mamba_pose_onnx_quickstart.md
Supplementary/mamba_pose_safety_test.md
Supplementary/mamba_pose_debug_checklist.md
```

### Models / Logs

```text
Log/models/small_nonzero_mamba_pose.onnx
Log/models/oversized_mamba_pose.onnx
Log/mamba_pose_train_data.csv
Log/mamba_pose_train_data_clean.csv
Log/mamba_pose_train_data_report.txt
```

---

## 7. Model Interface

### ONNX Input

Name:

```text
input
```

Shape:

```text
[sequence_length, 18]
```

Observed runtime shape:

```text
seq_len=10
feature_dim=18
flat_input_len=180
```

### Per-frame 18D Feature Order

```text
1. pos_x
2. pos_y
3. pos_z
4. rot_qx
5. rot_qy
6. rot_qz
7. rot_qw
8. vel_x
9. vel_y
10. vel_z
11. bias_g_x
12. bias_g_y
13. bias_g_z
14. bias_a_x
15. bias_a_y
16. bias_a_z
17. effective_feature_num
18. avg_residual
```

### ONNX Output

Name:

```text
output
```

Shape:

```text
[6]
```

Meaning:

```text
[d_roll, d_pitch, d_yaw, d_tx, d_ty, d_tz]
```

---

## 8. Verified ONNX Runtime State

ONNX Runtime C++ installed at:

```text
/opt/onnxruntime
```

Verified files:

```text
/opt/onnxruntime/include/onnxruntime_cxx_api.h
/opt/onnxruntime/lib/libonnxruntime.so
```

Verified `CMakeCache.txt`:

```text
ENABLE_ONNXRUNTIME:BOOL=ON
ONNXRUNTIME_INCLUDE_DIR:PATH=/opt/onnxruntime/include
ONNXRUNTIME_LIBRARY:FILEPATH=/opt/onnxruntime/lib/libonnxruntime.so
ONNXRUNTIME_ROOT:PATH=/opt/onnxruntime
```

Verified runtime linking:

```text
libonnxruntime.so.1.18.1 => /opt/onnxruntime/lib/libonnxruntime.so.1.18.1
```

Previously fixed CMake error:

```text
The plain signature for target_link_libraries has already been used with
the target "pose_compensator".
```

Fix:

```cmake
target_link_libraries(pose_compensator ${ONNXRUNTIME_LIBRARY})
```

instead of:

```cmake
target_link_libraries(pose_compensator PUBLIC ${ONNXRUNTIME_LIBRARY})
```

Reason:

`ament_target_dependencies(...)` used plain signature internally, so `pose_compensator` target must not mix keyword and plain signatures.

---

## 9. Verified Test Results

### 9.1 Normal Test

Config:

```text
config/mamba_pose_onnx_normal_test.yaml
```

Model:

```text
Log/models/small_nonzero_mamba_pose.onnx
```

Expected and verified:

```text
requested_backend=onnx
active_backend=onnx
model_loaded=true
session_ready=true
io_name_ready=true
inference_success=true
fallback=false
backend_status=io_name_ready
backend_error=none
inference_status=inference_success
seq_len=10
feature_dim=18
flat_input_len=180
output_dim=6
clamped=false
rejected=false
reject_reason=none
raw=[0.01, 0, 0, 0.01, 0, 0]
safe=[0.01, 0, 0, 0.01, 0, 0]
```

Conclusion:

```text
Normal ONNX Runtime inference path passed.
```

### 9.2 Oversized + Clamp Test

Config:

```text
config/mamba_pose_onnx_safety_test.yaml
```

Model:

```text
Log/models/oversized_mamba_pose.onnx
```

Expected and verified:

```text
raw=[0.3, 0, 0, 0.5, 0, 0]
safe=[0.1, 0, 0, 0.2, 0, 0]
clamped=true
rejected=false
reject_reason=none
```

Conclusion:

```text
Clamp path passed.
```

### 9.3 Oversized + Reject Test

Config:

```text
config/mamba_pose_onnx_reject_test.yaml
```

Model:

```text
Log/models/oversized_mamba_pose.onnx
```

Expected and verified:

```text
raw=[0.3, 0, 0, 0.5, 0, 0]
safe=[0.1, 0, 0, 0.2, 0, 0]
clamped=true
rejected=true
reject_reason=oversized_output
```

Conclusion:

```text
Reject path passed.
```

---

## 10. Current Data Export and Analysis State

Current CSV:

```text
/home/liu/fast_livo2/src/FAST-LIVO2/Log/mamba_pose_train_data.csv
```

Latest observed file status:

```text
-rw-rw-r-- 1 liu liu 49K 5月 17 00:14 /home/liu/fast_livo2/src/FAST-LIVO2/Log/mamba_pose_train_data.csv
```

Line count:

```text
131
```

So there are about 130 data samples plus one header row.

Observed header:

```text
timestamp,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,rot_w,vel_x,vel_y,vel_z,bias_g_x,bias_g_y,bias_g_z,bias_a_x,bias_a_y,bias_a_z,effective_feature_num,avg_residual,history_size,ready_flag,gt_pos_x,gt_pos_y,gt_pos_z,gt_rot_x,gt_rot_y,gt_rot_z,gt_rot_w
```

Observed sample properties:

```text
effective_feature_num is thousands, e.g. 8522, 8518, 9288.
avg_residual is positive, e.g. 0.0137, 0.0266, 0.0186.
history_size grows from 1 upward.
ready_flag changes from 0 to 1.
gt_* fields are nan.
```

Verified conclusions:

```text
CSV export works.
CSV header is correct.
CSV contains valid LIO values.
gt_* nan values are expected.
```

Latest script verification results:

```text
scripts/analyze_mamba_pose_train_data.py runs successfully.
Default report output works.
Default clean CSV output works with --save-clean.
Custom output path override works.
gt_* nan fields are not treated as invalid samples.
```

Latest analysis summary:

```text
total_rows=131
ready_flag_eq_1=128 (97.71%)
history_size: min=1, max=10, mean=9.653846
effective_feature_num: min=8518, max=11269, mean=10296.976923
avg_residual: min=0.013701, max=0.030151, mean=0.023705
effective_feature_num <= 0: 0
avg_residual < 0: 0
ready_flag == 0: 2
history_size < 10: 9
NaN / Inf in input feature fields: 1
valid_rows_after_filter=121 (92.37%)
clean_csv_line_count=122 including header
```

Important caveats:

```text
130 samples are enough to verify export and analysis scripts, but not enough to train a real Mamba model.
A longer rosbag run or more datasets will be needed for actual training.
Current gt_* fields are still nan, so the current clean CSV is a training-data base table, not yet a fully supervised training set.
```

---

## 11. Common Commands

### Build

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

### Check ONNX Runtime CMake State

```bash
grep -i "ONNX\|ONNXRUNTIME" /home/liu/fast_livo2/build/fast_livo/CMakeCache.txt
```

Expected:

```text
ONNXRUNTIME_INCLUDE_DIR=/opt/onnxruntime/include
ONNXRUNTIME_LIBRARY=/opt/onnxruntime/lib/libonnxruntime.so
ONNXRUNTIME_ROOT=/opt/onnxruntime
```

### Check Runtime Link

```bash
ldd /home/liu/fast_livo2/install/fast_livo/lib/fast_livo/fastlivo_mapping | grep onnx
```

Expected:

```text
libonnxruntime.so.1.18.1 => /opt/onnxruntime/lib/libonnxruntime.so.1.18.1
```

### Run Normal Test

```bash
cd /home/liu/fast_livo2
source /opt/ros/humble/setup.bash
source /home/liu/fast_livo2/install/setup.bash
export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH

ros2 launch fast_livo mapping_avia.launch.py \
  mamba_pose_params_file:=/home/liu/fast_livo2/src/FAST-LIVO2/config/mamba_pose_onnx_normal_test.yaml \
  2>&1 | tee ~/mamba_pose_normal_test.log
```

### Watch MambaPose Logs

```bash
tail -f ~/mamba_pose_normal_test.log | grep --line-buffered -i "MambaPose\|requested_backend\|active_backend\|model_loaded\|session_ready\|io_name_ready\|inference_success\|backend_status\|backend_error\|inference_status\|clamped\|rejected\|reject_reason\|raw=\|safe="
```

### Check Training CSV

```bash
ls -lh /home/liu/fast_livo2/src/FAST-LIVO2/Log/mamba_pose_train_data.csv
head -5 /home/liu/fast_livo2/src/FAST-LIVO2/Log/mamba_pose_train_data.csv
wc -l /home/liu/fast_livo2/src/FAST-LIVO2/Log/mamba_pose_train_data.csv
```

---

## 12. Current Bottlenecks

Current real bottlenecks:

```text
The current dataset is still small: about 130 data samples plus one header row.
The clean CSV has only 121 valid samples after filtering, which is enough for pipeline verification but not for real model training.
Current gt_* fields remain nan, so there is still no direct supervised pose-correction label source in the exported CSV.
The current clean CSV is suitable for later sequence slicing, but not yet sufficient by itself for final Mamba model training.
```

## 13. Current Next Task

Current next task:

> Stay out of the FAST-LIVO2 C++ main flow and continue on the data side.

Recommended next minimal engineering steps:

1. Collect more training CSV data from longer rosbag runs or more datasets.
2. Re-run `scripts/analyze_mamba_pose_train_data.py --save-clean` after each data collection round.
3. Use `Log/mamba_pose_train_data_clean.csv` as the base table for fixed-length sequence slicing, typically `T=10` or larger.
4. Design the next offline data-preparation step to build training sequences and later align them with ground-truth or pseudo-label targets.
5. Do not modify PoseCompensator inference logic, safety layer, or ONNX backend during this stage.

---

## 14. Do Not Modify

Do not modify:

```text
src/voxel_map.cpp
include/voxel_map.h
src/vio.cpp
include/vio.h
```

Also do not modify unless explicitly requested:

```text
src/pose_compensator.cpp
include/pose_compensator.h
src/LIVMapper.cpp
include/LIVMapper.h
CMakeLists.txt
launch/mapping_avia.launch.py
config/mamba_pose_onnx_normal_test.yaml
config/mamba_pose_onnx_safety_test.yaml
config/mamba_pose_onnx_reject_test.yaml
```

Do not:

```text
redesign the three innovation points
change ONNX backend
change PoseCompensator inference logic
change safety layer
change FAST-LIVO2 main flow
treat gt_* NaN fields as invalid samples
```

For the current task, only add the data-analysis script.
