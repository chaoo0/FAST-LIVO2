# Mamba Pose Safety Test

This note records the current ONNX correction and safety-layer validation flow for FAST-LIVO2.

Current main launch:

```text
launch/mapping_mid360.launch.py
```

The MID360 launch path already supports MambaPose parameter loading and automatic rosbag playback.

---

## Goal

Verify that:

1. a real ONNX model is loaded
2. inference returns a fixed non-zero 6D correction
3. the correction is decoded correctly
4. the correction enters the existing safety layer
5. `clamped`, `rejected`, `raw`, and `safe` match expectations

---

## Launch Arguments Relevant To This Test

`mapping_mid360.launch.py` currently supports:

```text
use_rviz
mamba_pose_params_file
play_bag
bag_path
bag_loop
bag_clock
```

Current default bag path:

```text
/home/liu/rosbags/mid360_fastlivo_mamba_20260519_211645
```

---

## Test Assets

Models:

- `Log/models/small_nonzero_mamba_pose.onnx`
- `Log/models/oversized_mamba_pose.onnx`

YAML:

- `config/mamba_pose_onnx_normal_test.yaml`
- `config/mamba_pose_onnx_safety_test.yaml`
- `config/mamba_pose_onnx_reject_test.yaml`

Important:

> These YAML files are only for ONNX / safety testing. Do not use them for formal training-data export.

Formal export must use:

```text
config/mamba_pose_train_export_dummy.yaml
```

Reason:

> The test models output fixed corrections. Using them during formal CSV export would pollute the original FAST-LIVO2 state sequence.

---

## Output Convention

The ONNX test models output a fixed 6D vector in this order:

```text
[d_roll, d_pitch, d_yaw, d_tx, d_ty, d_tz]
```

### Normal Mode

`small_nonzero_mamba_pose.onnx` outputs:

```text
[0.01, 0.0, 0.0, 0.01, 0.0, 0.0]
```

This should pass through the default safety limits without clamp or reject.

### Oversized Mode

`oversized_mamba_pose.onnx` outputs:

```text
[0.30, 0.0, 0.0, 0.50, 0.0, 0.0]
```

With current limits:

- `max_rotation_correction_rad = 0.10`
- `max_translation_correction_m = 0.20`

this should trigger clamp.

If:

```text
mamba_pose/reject_oversized_output = true
```

then the same oversized output should be rejected.

---

## Current Verified Status

The same safety behavior that was first verified in the Avia path has now been synchronized to MID360.

MID360 normal-path logs have already been confirmed with:

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

Conclusion:

```text
MID360 normal / clamp / reject safety behavior is considered verified.
```

---

## Per-Round YAML Mapping

### Round 1: Normal Non-Zero Output

Use:

```text
config/mamba_pose_onnx_normal_test.yaml
```

Confirm:

- `mamba_pose/model_path` points to `small_nonzero_mamba_pose.onnx`
- `mamba_pose/reject_oversized_output=false`

### Round 2: Oversized Output + Clamp

Use:

```text
config/mamba_pose_onnx_safety_test.yaml
```

Confirm:

- `mamba_pose/model_path` points to `oversized_mamba_pose.onnx`
- `mamba_pose/reject_oversized_output=false`

### Round 3: Oversized Output + Reject

Use:

```text
config/mamba_pose_onnx_reject_test.yaml
```

Confirm:

- `mamba_pose/model_path` points to `oversized_mamba_pose.onnx`
- `mamba_pose/reject_oversized_output=true`

---

## What To Watch In Logs

The most useful `MambaPose` fields are:

- `requested_backend`
- `active_backend`
- `model_loaded`
- `session_ready`
- `io_name_ready`
- `inference_success`
- `backend_status`
- `backend_error`
- `inference_status`
- `raw`
- `safe`
- `clamped`
- `rejected`
- `reject_reason`

---

## How To Judge The Result

### A. Non-Zero Output Entered The Safe Correction Path

You should see:

1. `requested_backend=onnx`
2. `active_backend=onnx`
3. `model_loaded=true`
4. `inference_success=true`
5. `raw` is non-zero
6. `safe` is non-zero
7. `clamped=false`
8. `rejected=false`

### B. Correction Was Clamped

Use the oversized model with:

```text
mamba_pose/reject_oversized_output=false
```

You should see:

1. `raw` contains the large configured values
2. `safe` is smaller than `raw`
3. `clamped=true`
4. `rejected=false`

### C. Correction Was Rejected

Use the oversized model with:

```text
mamba_pose/reject_oversized_output=true
```

You should see:

1. `raw` contains the large configured values
2. `clamped=true`
3. `rejected=true`
4. `reject_reason=oversized_output`

---

## Recommended MID360 Command Pattern

Normal round:

```bash
ros2 launch fast_livo mapping_mid360.launch.py \
  mamba_pose_params_file:=/home/liu/fast_livo2/src/FAST-LIVO2/config/mamba_pose_onnx_normal_test.yaml \
  play_bag:=true \
  bag_path:=/home/liu/rosbags/mid360_fastlivo_mamba_20260519_211645 \
  bag_loop:=false \
  bag_clock:=true
```

Clamp round:

```bash
ros2 launch fast_livo mapping_mid360.launch.py \
  mamba_pose_params_file:=/home/liu/fast_livo2/src/FAST-LIVO2/config/mamba_pose_onnx_safety_test.yaml \
  play_bag:=true \
  bag_path:=/home/liu/rosbags/mid360_fastlivo_mamba_20260519_211645 \
  bag_loop:=false \
  bag_clock:=true
```

Reject round:

```bash
ros2 launch fast_livo mapping_mid360.launch.py \
  mamba_pose_params_file:=/home/liu/fast_livo2/src/FAST-LIVO2/config/mamba_pose_onnx_reject_test.yaml \
  play_bag:=true \
  bag_path:=/home/liu/rosbags/mid360_fastlivo_mamba_20260519_211645 \
  bag_loop:=false \
  bag_clock:=true
```
