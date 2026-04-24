# Mamba Pose ONNX Quickstart

This note is for the first end-to-end ONNX smoke test in FAST-LIVO2.

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

Use your normal launch flow, but point it at the ONNX test parameter file.

If your launch setup already merges multiple YAML files, keep your existing lidar/camera config and add the ONNX test file on top. The exact command depends on your local launch method.

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
