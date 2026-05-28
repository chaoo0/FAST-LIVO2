# Mamba Pose ONNX Quickstart

This note is for the first end-to-end ONNX smoke test in FAST-LIVO2.

Current main launch:

```text
launch/mapping_mid360.launch.py
```

Current verified no-op baseline ONNX model:

```text
Log/models/static_zero_mamba_pose.onnx
```

Current runtime-ready raw-input ONNX model:

```text
Log/models/static_zero_mamba_pose_raw_input.onnx
```

## Goal

Verify the following path once, with the safest possible model:

1. `backend_type=onnx`
2. the ONNX model loads successfully
3. input and output names match
4. inference returns once per ready frame
5. the 6D correction reaches the existing safety layer
6. the main FAST-LIVO2 pipeline stays stable

## Files Added For The Smoke Test

- `scripts/export_dummy_mamba_pose_onnx.py`
- `config/mamba_pose_onnx_test.yaml`

## Minimal ONNX Contract

The provided export script generates the simplest model expected by the current runtime path:

- single input
- single output
- input name: `input`
- output name: `output`
- input shape: `[T, 18]`
- output shape: `[6]`
- output meaning:
  `[d_roll, d_pitch, d_yaw, d_tx, d_ty, d_tz]`
- output value:
  always zero

This model is not for correction quality. It is only for the first integration test.

## Step 1: Export The Dummy ONNX Model

From the repository root:

```bash
python3 scripts/export_dummy_mamba_pose_onnx.py
```

Default export path:

```text
Log/models/dummy_mamba_pose.onnx
```

If you want a custom path:

```bash
python3 scripts/export_dummy_mamba_pose_onnx.py \
  --output /absolute/path/to/dummy_mamba_pose.onnx \
  --input-name input \
  --output-name output
```

## Step 2: Prepare ROS2 Parameters

Start from:

```text
config/mamba_pose_onnx_test.yaml
```

Update at least:

- `mamba_pose/model_path`
- optionally `mamba_pose/export_train_data_path`

The sample file already uses:

- `backend_type=onnx`
- `onnx_input_name=input`
- `onnx_output_name=output`
- conservative safety limits
- `debug_log_en=true`

## Step 3: Launch FAST-LIVO2

Use the current MID360 launch flow and point it at the ONNX test parameter file.

Example:

```bash
ros2 launch fast_livo mapping_mid360.launch.py \
  mamba_pose_params_file:=/home/liu/fast_livo2/src/FAST-LIVO2/config/mamba_pose_onnx_test.yaml \
  play_bag:=true \
  bag_path:=/home/liu/rosbags/mid360_fastlivo_mamba_20260519_211645 \
  bag_loop:=false \
  bag_clock:=true
```

## What To Watch In Logs

The most useful `MambaPose` fields during the first smoke test are:

- `requested_backend`
- `active_backend`
- `model_loaded`
- `session_ready`
- `io_name_ready`
- `inference_success`
- `backend_status`
- `inference_status`
- `seq_len`
- `feature_dim`
- `flat_input_len`
- `output_dim`
- `raw`
- `safe`

## First-Test Success Criteria

The first ONNX smoke test is successful if you see all of the following:

1. `requested_backend=onnx`
2. `active_backend=onnx`
3. `model_loaded=true`
4. `session_ready=true`
5. `io_name_ready=true`
6. `inference_success=true` after the history becomes ready
7. `output_dim=6`
8. `raw=[0, 0, 0, 0, 0, 0]` or numerically equivalent zero output
9. FAST-LIVO2 continues to run normally without instability

## Common Failure Modes

- `model_loaded=false`
  `model_path` is wrong or the model file does not exist.

- `session_ready=false`
  ONNX Runtime could not create the session or the model is invalid.

- `io_name_ready=false`
  The configured input/output names do not match the model.

- `inference_success=false`
  Input shape, output shape, or runtime execution still does not match the current backend assumptions.

- `backend_status=onnxruntime_not_compiled`
  FAST-LIVO2 was built without ONNX Runtime support. Rebuild with ONNX Runtime available to CMake.

## Recommended First Model

For the first live integration, do not start with a trained model.

Start with the provided zero-output dummy model because it lets you verify:

- model loading
- input/output naming
- tensor shape compatibility
- runtime inference execution
- safety-layer pass-through

Only after that should you swap in a real model.

## Important Separation

`config/mamba_pose_onnx_test.yaml` is only for ONNX smoke testing.

Do not use ONNX test YAML for formal training-data export.

Formal export should use:

```text
config/mamba_pose_train_export_dummy.yaml
```

## Static Zero Baseline Model

The current offline training side can now export a first no-op baseline model:

```text
Log/models/static_zero_mamba_pose.onnx
```

This model is only for the current stationary-rosbag no-op baseline. It validates the ONNX interface and deployment loop, not real dynamic pose correction ability.

### ONNX Interface

- input name: `input`
- output name: `output`
- input shape: `[10, 18]`
- output shape: `[6]`
- output meaning:
  `[d_roll, d_pitch, d_yaw, d_tx, d_ty, d_tz]`

### How To Produce It

```bash
python3 scripts/train_mamba_pose_static_zero.py
```

### ONNX Smoke Test

The training script already performs an automatic smoke test when both `onnx` and `onnxruntime` are installed.

Equivalent manual check:

```bash
python3 - <<'PY'
import numpy as np
import onnx
import onnxruntime as ort

path = "Log/models/static_zero_mamba_pose.onnx"
onnx_model = onnx.load(path)
onnx.checker.check_model(onnx_model)
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
output = session.run(["output"], {"input": np.zeros((10, 18), dtype=np.float32)})[0]
print("output_shape=", output.shape)
print("output=", output)
PY
```

### Runtime Loading Note

For FAST-LIVO2 runtime verification, keep:

- `mamba_pose/backend_type=onnx`
- `mamba_pose/onnx_input_name=input`
- `mamba_pose/onnx_output_name=output`

and point:

```text
mamba_pose/model_path = /home/liu/fast_livo2/src/FAST-LIVO2/Log/models/static_zero_mamba_pose_raw_input.onnx
```

Expected stationary no-op behavior:

- `active_backend=onnx`
- `model_loaded=true`
- `session_ready=true`
- `io_name_ready=true`
- `inference_success=true`
- `raw` close to zero
- `safe` close to zero

## Raw-Input Runtime-Ready ONNX

The normalized-input model above is useful for offline comparison, but it should not be wired directly into the current FAST-LIVO2 C++ ONNX backend.

Reason:

> FAST-LIVO2 currently sends raw basic18 features into ONNX at runtime, not normalized features.

The runtime-ready export is:

```text
Log/models/static_zero_mamba_pose_raw_input.onnx
```

Its interface is still:

- input name: `input`
- output name: `output`
- input shape: `[10, 18]`
- output shape: `[6]`

but inside the ONNX graph it first computes:

```text
X_normalized = (input - mean) / std_safe
```

and only then forwards the normalized tensor into the trained static-zero MLP.

### How To Produce The Raw-Input ONNX

```bash
python3 scripts/export_static_zero_mamba_pose_raw_input_onnx.py
```

### Raw vs Normalized ONNX Comparison

The export script compares:

- `static_zero_mamba_pose_raw_input.onnx(raw X[i])`
- `static_zero_mamba_pose.onnx(X_normalized[i])`

Expected:

- the two outputs should be very close
- the output should still be close to zero on the current stationary samples
