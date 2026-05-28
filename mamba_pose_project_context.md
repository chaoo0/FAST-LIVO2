# FAST-LIVO2 + MambaPose Project Context

## 1. Project Identity

This repository is being used for the current research prototype:

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
```

Current ONNX Runtime path:

```text
/opt/onnxruntime
```

Current main launch file:

```text
/home/liu/fast_livo2/src/FAST-LIVO2/launch/mapping_mid360.launch.py
```

Current training CSV path:

```text
/home/liu/fast_livo2/src/FAST-LIVO2/Log/mamba_pose_train_data.csv
```

---

## 2. Current Scope

The project is still focused only on:

> **Innovation Point 1: Mamba-based temporal pose compensation inside FAST-LIVO2**

Do not restart the design from scratch.

Do not expand into:

- point cloud / local map temporal feature enhancement
- implicit loop closure / global memory

unless explicitly requested.

Current intended runtime path:

```text
FAST-LIVO2 handleLIO()
        ↓
StateEstimation()
        ↓
PoseCompensator
        ↓
history sequence
        ↓
ONNX Runtime backend
        ↓
6D correction
        ↓
safety layer
        ↓
derived data rebuild
        ↓
UpdateVoxelMap()
```

This engineering chain has already been verified.

---

## 3. Current Main Launch

The current main launch is:

```text
launch/mapping_mid360.launch.py
```

It already supports:

- MID360 parameter loading
- MambaPose parameter loading
- optional RViz
- automatic rosbag playback

Current launch arguments:

```text
use_rviz
mamba_pose_params_file
play_bag
bag_path
bag_loop
bag_clock
```

Argument meaning:

- `play_bag`: whether to automatically run `ros2 bag play`
- `bag_path`: rosbag path to play
- `bag_loop`: whether to add `--loop`
- `bag_clock`: whether to add `--clock`

Current default bag path:

```text
/home/liu/rosbags/mid360_fastlivo_mamba_20260519_211645
```

---

## 4. Verified Engineering State

The following items are already verified:

- `PoseCompensator` enters the `handleLIO()` path.
- MambaPose parameters load correctly from ROS2 YAML.
- ONNX Runtime C++ is found by CMake.
- `fastlivo_mapping` links to `libonnxruntime.so.1.18.1`.
- the real ONNX backend runs successfully
- the safety layer behavior is correct
- derived data rebuild after compensation is active
- training CSV export works

Verified runtime chain:

```text
FAST-LIVO2 handleLIO()
→ StateEstimation()
→ PoseCompensator
→ history sequence
→ ONNX Runtime backend
→ 6D correction
→ safety layer
→ derived data rebuild
→ UpdateVoxelMap()
```

---

## 5. ONNX Runtime State

ONNX Runtime C++ is installed at:

```text
/opt/onnxruntime
```

Verified files:

```text
/opt/onnxruntime/include/onnxruntime_cxx_api.h
/opt/onnxruntime/lib/libonnxruntime.so
```

Verified CMake state:

```text
ENABLE_ONNXRUNTIME:BOOL=ON
ONNXRUNTIME_INCLUDE_DIR:PATH=/opt/onnxruntime/include
ONNXRUNTIME_LIBRARY:FILEPATH=/opt/onnxruntime/lib/libonnxruntime.so
ONNXRUNTIME_ROOT:PATH=/opt/onnxruntime
```

Verified runtime link:

```text
libonnxruntime.so.1.18.1 => /opt/onnxruntime/lib/libonnxruntime.so.1.18.1
```

---

## 6. Current Test YAML Rules

The ONNX / safety test YAML files are only for validation.

### 6.1 Test-Only YAML

`config/mamba_pose_onnx_normal_test.yaml`

- backend: `onnx`
- model: `small_nonzero_mamba_pose.onnx`
- fixed output: `[0.01, 0, 0, 0.01, 0, 0]`
- use only for normal-path testing

`config/mamba_pose_onnx_safety_test.yaml`

- backend: `onnx`
- model: `oversized_mamba_pose.onnx`
- fixed output: `[0.3, 0, 0, 0.5, 0, 0]`
- use only for clamp-path testing

`config/mamba_pose_onnx_reject_test.yaml`

- backend: `onnx`
- model: `oversized_mamba_pose.onnx`
- `reject_oversized_output=true`
- use only for reject-path testing

These three files are **not** for formal training-data export.

### 6.2 Formal Export YAML

Formal training-data export should use:

```text
config/mamba_pose_train_export_dummy.yaml
```

Required settings:

```text
mamba_pose/enabled=true
mamba_pose/backend_type=dummy
mamba_pose/model_path=""
mamba_pose/history_len=10
mamba_pose/min_ready_frames=10
mamba_pose/export_train_data_en=true
mamba_pose/export_train_data_path=/home/liu/fast_livo2/src/FAST-LIVO2/Log/mamba_pose_train_data.csv
```

Reason:

> When exporting formal training data, do not use ONNX test models, otherwise the fixed small or oversized corrections will contaminate the original FAST-LIVO2 state history.

---

## 7. Verified ONNX / Safety Results

The normal / clamp / reject tests were first validated in the Avia flow and have now been synchronized to the MID360 launch flow.

MID360 normal test has been explicitly confirmed with logs including:

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

Clamp-path expectation and verified behavior:

```text
raw=[0.3, 0, 0, 0.5, 0, 0]
safe=[0.1, 0, 0, 0.2, 0, 0]
clamped=true
rejected=false
reject_reason=none
```

Reject-path expectation and verified behavior:

```text
raw=[0.3, 0, 0, 0.5, 0, 0]
safe=[0.1, 0, 0, 0.2, 0, 0]
clamped=true
rejected=true
reject_reason=oversized_output
```

Current conclusion:

```text
MID360 normal / clamp / reject capability is considered verified.
```

---

## 8. Training Data Export State

Current exported CSV:

```text
Log/mamba_pose_train_data.csv
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

Notes:

- `gt_*` fields are allowed to be `nan`
- `gt_*` fields do not participate in the current cleaning filter
- do not use stale sample-count notes from older documents as the current truth

---

## 9. Completed Offline Data Scripts

### 9.1 CSV Analysis / Cleaning

Implemented script:

```text
scripts/analyze_mamba_pose_train_data.py
```

Purpose:

- read `Log/mamba_pose_train_data.csv`
- filter abnormal rows
- output `Log/mamba_pose_train_data_clean.csv`
- output `Log/mamba_pose_train_data_report.txt`

Current verified result:

```text
raw total_rows = 1311
cleaned_row_count = 1302
```

Current cleaning rules:

- `ready_flag == 1`
- `history_size >= 10`
- `effective_feature_num > 0`
- `avg_residual >= 0`
- no `NaN` / `Inf` in input feature fields
- `gt_*` may be `NaN` and must not be used for filtering

### 9.2 Short-Gap Interpolation

Implemented script:

```text
scripts/interpolate_mamba_pose_time_gaps.py
```

Purpose:

- read `Log/mamba_pose_train_data_clean.csv`
- interpolate only short timestamp gaps
- output `Log/mamba_pose_train_data_interpolated.csv`
- output `Log/mamba_pose_train_data_interpolated_report.txt`

Current verified interpolation result:

```text
original_rows = 1302
output_rows = 1307
inserted_rows = 5
interpolatable_gap_count = 2
actually_interpolated_gap_count = 2
max_gap_before = 0.299603
max_gap_after = 0.100726
basic18_nan_or_inf_count = 0
```

Key behavior:

- do not interpolate `dt <= 0`
- do not blindly average all discontinuities
- only interpolate `max_continuous_gap < dt <= max_interpolate_gap`
- keep long gaps as true segment boundaries
- use normalized quaternion interpolation
- do not generate `y` labels
- do not use `gt_*`

### 9.3 Fixed-Length Sequence Builder

Implemented script:

```text
scripts/build_mamba_pose_sequences.py
```

Purpose:

- read a clean or interpolated CSV
- split segments using timestamp continuity
- build sliding windows with `T=10`
- output sequence NPZ and TXT report

Current basic18 feature order must remain:

```text
pos_x,pos_y,pos_z,
rot_x,rot_y,rot_z,rot_w,
vel_x,vel_y,vel_z,
bias_g_x,bias_g_y,bias_g_z,
bias_a_x,bias_a_y,bias_a_z,
effective_feature_num,
avg_residual
```

Current verified sequence comparison:

```text
clean baseline:
  X shape = [1281, 10, 18]
  skipped_by_time_gap = 12
  segment_count = 3

interpolated:
  X shape = [1298, 10, 18]
  skipped_by_time_gap = 0
  segment_count = 1
```

Current main sequence artifact:

```text
Log/mamba_pose_sequences_T10_interpolated.npz
Log/mamba_pose_sequences_T10_interpolated_report.txt
```

### 9.4 Feature Normalization Statistics

Implemented script:

```text
scripts/compute_mamba_pose_feature_norm.py
```

Purpose:

- read `Log/mamba_pose_sequences_T10_interpolated.npz`
- compute per-feature normalization statistics for basic18
- output `Log/mamba_pose_feature_norm_T10_interpolated.npz`
- output `Log/mamba_pose_feature_norm_T10_interpolated_report.txt`
- optionally output `Log/mamba_pose_sequences_T10_interpolated_normalized.npz`

Current verified result:

```text
input X shape = [1298, 10, 18]
X_global_nan_count = 0
X_global_inf_count = 0
std_lt_epsilon_features = none
normalized_feature_mean_range ≈ [0, 0]
normalized_feature_std_range ≈ [1, 1]
```

Important note:

> Later training and ONNX-side preprocessing must reuse the same `mean` and `std_safe` saved by the norm NPZ.

### 9.5 Static Zero-Label Baseline Dataset

Implemented script:

```text
scripts/build_mamba_pose_static_zero_label_dataset.py
```

Purpose:

- read `Log/mamba_pose_sequences_T10_interpolated_normalized.npz`
- read `Log/mamba_pose_feature_norm_T10_interpolated.npz`
- build a first `static_zero` label dataset
- output `Log/mamba_pose_static_zero_label_dataset_T10.npz`
- output `Log/mamba_pose_static_zero_label_dataset_T10_report.txt`

Current verified result:

```text
X_normalized shape = [9919, 10, 18]
y shape = [9919, 6]
y is all zero
X NaN count = 0
X Inf count = 0
y NaN count = 0
y Inf count = 0
label_type = static_zero
```

Current label definition:

```text
y = [d_roll, d_pitch, d_yaw, d_tx, d_ty, d_tz] = [0, 0, 0, 0, 0, 0]
```

Important limitation:

> This is only a stationary-rosbag no-op baseline label. It is useful for validating training, dataloader, loss, ONNX export, and FAST-LIVO2 inference-loop integration later, but it is not the final real pose correction label and cannot prove dynamic compensation ability.

### 9.6 Static Zero-Label No-Op Baseline Training

Implemented script:

```text
scripts/train_mamba_pose_static_zero.py
```

Purpose:

- read `Log/mamba_pose_static_zero_label_dataset_T10.npz`
- train a small `flatten + MLP -> [6]` temporal baseline
- save best checkpoint
- export a FAST-LIVO2-compatible ONNX model
- run ONNX smoke test when `onnx` and `onnxruntime` are available

Current output files:

```text
Log/models/static_zero_mamba_pose.pt
Log/models/static_zero_mamba_pose.onnx
Log/models/static_zero_mamba_pose_train_log.csv
Log/models/static_zero_mamba_pose_train_report.txt
```

Current verified training result:

```text
dataset X shape = [9919, 10, 18]
dataset y shape = [9919, 6]
train_size = 7935
val_size = 1984
epochs = 30
batch_size = 128
best_val_loss = 6.59830856963503e-07
final_val_loss = 6.59830856963503e-07
pred_abs_max = 0.001578
pred_abs_mean = 0.000474
```

Current verified ONNX result:

```text
onnx_export_status = exported
onnx_smoke_test_status = passed
onnx_input_name = input
onnx_output_name = output
onnx_input_shape = [10, 18]
onnx_output_shape = [6]
zero_input_output_abs_max = 0.008137
zero_input_output_abs_mean = 0.003191
```

Current interpretation:

> The no-op baseline training and ONNX export loop is now verified end to end, but only for the current `static_zero` target. It validates the training / export / runtime-interface loop, not real dynamic pose correction ability.

### 9.7 Runtime-Ready Raw-Input ONNX Export

Implemented script:

```text
scripts/export_static_zero_mamba_pose_raw_input_onnx.py
```

Purpose:

- load `Log/models/static_zero_mamba_pose.pt`
- load `Log/mamba_pose_feature_norm_T10_interpolated.npz`
- embed `mean` / `std_safe` into an ONNX export wrapper
- export a raw-input ONNX model compatible with the current FAST-LIVO2 C++ backend
- compare raw-input ONNX(raw X) against normalized-input ONNX(X_normalized)

Current output files:

```text
Log/models/static_zero_mamba_pose_raw_input.onnx
Log/models/static_zero_mamba_pose_raw_input_onnx_report.txt
```

Current verified result:

```text
input shape = [10, 18]
output shape = [6]
onnx_checker_status = passed
onnxruntime_smoke_test_status = passed
raw_onnx_output_abs_max = 0.000628
raw_onnx_output_abs_mean = 0.000308
normalized_onnx_output_abs_max = 0.000624
normalized_onnx_output_abs_mean = 0.000304
output_diff_abs_max_between_raw_and_normalized_onnx = 0.000010
output_diff_abs_mean_between_raw_and_normalized_onnx = 0.000004
```

Current interpretation:

> `static_zero_mamba_pose_raw_input.onnx` is the correct model to connect back into FAST-LIVO2, because the current C++ ONNX path sends raw basic18 features, not normalized features.

### 9.8 Runtime Observe-Only Safety Switch

Current runtime finding:

> `static_zero_mamba_pose_raw_input.onnx` can now be loaded and inferred successfully inside FAST-LIVO2, but the current `static_zero` no-op baseline is not stable enough to be directly closed-loop applied to `_state`.

Observed runtime problem:

- ONNX loading and inference succeed
- `raw` correction may start small and later drift to very large values
- if `reject_oversized_output=false`, the safety layer only clamps to bounded `safe`
- `rejected=false` still allows the clamped `safe` correction to be written back
- this can push FAST-LIVO2 state and map far away even though the ONNX path itself is technically alive

Minimal C++ fix now applied:

- new runtime parameter: `mamba_pose/apply_correction_en`
- compatible alias: `mamba_pose.apply_correction_en`
- default value: `false`
- modified files:
  - `include/LIVMapper.h`
  - `src/LIVMapper.cpp`
- runtime observe-only config:
  - `config/mamba_pose_static_zero_runtime_observe_only.yaml`

Current runtime control split:

- `mamba_pose/enabled=true` controls:
  - history buffering
  - ready-state checks
  - `pose_compensator_.compensate(_state)`
  - ONNX inference
  - raw / safe / clamped / rejected debug logs
- `mamba_pose/apply_correction_en=true` additionally allows:
  - `_state = compensated_state`
  - `voxelmap_manager->state_ = _state`

Current behavior matrix:

| `mamba_pose/enabled` | `mamba_pose/apply_correction_en` | Behavior |
| --- | --- | --- |
| `false` | `false` or `true` | MambaPose path stays off; no history, no inference, no correction logs |
| `true` | `false` | observe-only mode; history and inference still run, logs still print, but corrected state is not written back |
| `true` | `true` | previous full apply behavior; corrected state is written back to `_state` and `voxelmap_manager->state_` |

Current debug fields now also include:

```text
applied=true/false
apply_correction_en=true/false
```

Current interpretation:

> The raw-input ONNX runtime path is now verified separately from the state-writeback path. This lets us observe unstable `static_zero` model outputs safely before enabling true correction application.

---

## 10. Current Data Scale And Artifacts

Current verified offline-data scale:

```text
current latest rosbag state = stationary
interpolated CSV rows = 9928
interpolated T=10 sequence count = 9919
normalized sequence count = 9919
static zero-label dataset count = 9919
feature_dim = 18
label_dim = 6
```

Current main offline artifacts:

```text
Log/mamba_pose_train_data_interpolated.csv
Log/mamba_pose_sequences_T10_interpolated.npz
Log/mamba_pose_feature_norm_T10_interpolated.npz
Log/mamba_pose_sequences_T10_interpolated_normalized.npz
Log/mamba_pose_static_zero_label_dataset_T10.npz
Log/models/static_zero_mamba_pose.pt
Log/models/static_zero_mamba_pose.onnx
Log/models/static_zero_mamba_pose_raw_input.onnx
config/mamba_pose_static_zero_runtime_observe_only.yaml
```

Current interpretation:

- the X-side data preparation chain is now available from CSV export to normalized sequence NPZ
- the first baseline `y` dataset is now available as `static_zero`
- the first no-op baseline model and ONNX artifact are now available
- the runtime-ready raw-input ONNX wrapper is now available
- the runtime path now has an observe-only switch so inference can be inspected without modifying `_state`
- current artifacts still do not define the final real dynamic correction label `y`
- `gt_*` fields remain reserved and should not be treated as ready-made labels

---

## 11. Current Bottlenecks

Current bottlenecks are now on the training-target side rather than the X-side preprocessing side:

- there is still no finalized real dynamic correction-label definition `y`
- the current `static_zero` dataset and model only validate a stationary no-op baseline
- later training and deployment must keep the same normalization parameters between Python training and ONNX-side inference preprocessing
- the exported raw-input no-op ONNX model still needs observe-only runtime verification before any future correction application is enabled

---

## 12. Next Offline Task

Current next recommended task:

> Use `Log/models/static_zero_mamba_pose_raw_input.onnx` with `config/mamba_pose_static_zero_runtime_observe_only.yaml` and verify the runtime ONNX path in observe-only mode before allowing any correction writeback.

Recommended next steps:

1. Run FAST-LIVO2 with `config/mamba_pose_static_zero_runtime_observe_only.yaml`.
2. Confirm runtime logs show `active_backend=onnx`, `model_loaded=true`, `session_ready=true`, `io_name_ready=true`, and `inference_success=true`.
3. Confirm the new log fields show `apply_correction_en=false` and `applied=false`, even after the model becomes ready and inference starts running.
4. Use this observe-only round to inspect whether `raw` / `safe` remain near zero or drift over time without risking `_state` corruption.
5. After observe-only runtime behavior is understood, decide whether to tighten rejection policy or move directly to a real correction-label `y` design for dynamic training data.

---

## 13. Key Files

### Launch

```text
launch/mapping_mid360.launch.py
launch/mapping_avia.launch.py
```

### Config

```text
config/mamba_pose_onnx_normal_test.yaml
config/mamba_pose_onnx_safety_test.yaml
config/mamba_pose_onnx_reject_test.yaml
config/mamba_pose_train_export_dummy.yaml
config/mamba_pose_static_zero_runtime_observe_only.yaml
```

### Offline Scripts

```text
scripts/analyze_mamba_pose_train_data.py
scripts/interpolate_mamba_pose_time_gaps.py
scripts/build_mamba_pose_sequences.py
scripts/compute_mamba_pose_feature_norm.py
scripts/build_mamba_pose_static_zero_label_dataset.py
scripts/train_mamba_pose_static_zero.py
scripts/export_static_zero_mamba_pose_raw_input_onnx.py
```

### Runtime Integration Files

```text
include/pose_compensator.h
src/pose_compensator.cpp
include/LIVMapper.h
src/LIVMapper.cpp
CMakeLists.txt
```

### Documents

```text
AGENTS.md
mamba_pose_project_context.md
Supplementary/mamba_pose_debug_checklist.md
Supplementary/mamba_pose_onnx_quickstart.md
Supplementary/mamba_pose_safety_test.md
```

---

## 14. Common Commands

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
grep -i "ONNX\\|ONNXRUNTIME" /home/liu/fast_livo2/build/fast_livo/CMakeCache.txt
```

### Check Runtime Link

```bash
ldd /home/liu/fast_livo2/install/fast_livo/lib/fast_livo/fastlivo_mapping | grep onnx
```

### Run MID360 Normal Test

```bash
cd /home/liu/fast_livo2
source /opt/ros/humble/setup.bash
source /home/liu/fast_livo2/install/setup.bash
export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH

ros2 launch fast_livo mapping_mid360.launch.py \
  mamba_pose_params_file:=/home/liu/fast_livo2/src/FAST-LIVO2/config/mamba_pose_onnx_normal_test.yaml \
  play_bag:=true \
  bag_path:=/home/liu/rosbags/mid360_fastlivo_mamba_20260519_211645 \
  bag_loop:=false \
  bag_clock:=true
```

### Export Formal Training Data

```bash
cd /home/liu/fast_livo2
source /opt/ros/humble/setup.bash
source /home/liu/fast_livo2/install/setup.bash

ros2 launch fast_livo mapping_mid360.launch.py \
  mamba_pose_params_file:=/home/liu/fast_livo2/src/FAST-LIVO2/config/mamba_pose_train_export_dummy.yaml
```

### Analyze And Clean CSV

```bash
python3 scripts/analyze_mamba_pose_train_data.py --save-clean
```

### Interpolate Short Gaps

```bash
python3 scripts/interpolate_mamba_pose_time_gaps.py
```

### Build T=10 Sequences From Interpolated CSV

```bash
python3 scripts/build_mamba_pose_sequences.py \
  --input Log/mamba_pose_train_data_interpolated.csv \
  --output Log/mamba_pose_sequences_T10_interpolated.npz \
  --report Log/mamba_pose_sequences_T10_interpolated_report.txt
```

### Compute Feature Normalization

```bash
python3 scripts/compute_mamba_pose_feature_norm.py
```

### Compute Feature Normalization And Save Normalized Sequence NPZ

```bash
python3 scripts/compute_mamba_pose_feature_norm.py --save-normalized
```

### Build Static Zero-Label Dataset

```bash
python3 scripts/build_mamba_pose_static_zero_label_dataset.py
```

### Train Static Zero No-Op Baseline And Export ONNX

```bash
python3 scripts/train_mamba_pose_static_zero.py
```

### Export Runtime-Ready Raw-Input ONNX

```bash
python3 scripts/export_static_zero_mamba_pose_raw_input_onnx.py
```

### Run Runtime Observe-Only Verification

```bash
cd /home/liu/fast_livo2
source /opt/ros/humble/setup.bash
source /home/liu/fast_livo2/install/setup.bash
export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH

ros2 launch fast_livo mapping_mid360.launch.py \
  mamba_pose_params_file:=/home/liu/fast_livo2/src/FAST-LIVO2/config/mamba_pose_static_zero_runtime_observe_only.yaml \
  play_bag:=true \
  bag_path:=/home/liu/rosbags/mid360_fastlivo_mamba_20260519_211645 \
  bag_loop:=false \
  bag_clock:=true
```

---

## 15. Do Not Modify

Do not modify unless explicitly requested:

```text
src/voxel_map.cpp
include/voxel_map.h
src/vio.cpp
include/vio.h
```

Also avoid modifying unless the task truly requires it:

```text
src/pose_compensator.cpp
include/pose_compensator.h
src/LIVMapper.cpp
include/LIVMapper.h
CMakeLists.txt
launch/mapping_avia.launch.py
```

Do not:

- redesign the project scope
- expand to innovation points 2 or 3
- refactor the ONNX backend
- change the safety layer logic
- change the PoseCompensator inference logic
- modify the FAST-LIVO2 C++ main flow for the current offline-data tasks

---

## 16. Recommended First Read For A New Codex Window

When a new Codex window starts in this repository, it should read these files first:

1. `AGENTS.md`
2. `mamba_pose_project_context.md`
3. `Supplementary/mamba_pose_debug_checklist.md`
4. `Supplementary/mamba_pose_onnx_quickstart.md`
5. `Supplementary/mamba_pose_safety_test.md`

After that, if the task is about offline data preparation, check:

```text
scripts/analyze_mamba_pose_train_data.py
scripts/interpolate_mamba_pose_time_gaps.py
scripts/build_mamba_pose_sequences.py
scripts/compute_mamba_pose_feature_norm.py
scripts/build_mamba_pose_static_zero_label_dataset.py
scripts/train_mamba_pose_static_zero.py
scripts/export_static_zero_mamba_pose_raw_input_onnx.py
config/mamba_pose_train_export_dummy.yaml
```
