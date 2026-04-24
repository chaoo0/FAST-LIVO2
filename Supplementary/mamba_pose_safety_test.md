# Mamba Pose Safety Test

This note is for validating the correction decoding path and the existing safety layer with a real ONNX inference call.

## Goal

Verify, with the smallest possible test assets, that:

1. a real ONNX model is loaded
2. inference returns a non-zero 6D correction
3. the correction is decoded correctly
4. the correction enters the existing safety layer
5. `clamped`, `rejected`, `raw`, and `safe` logs match expectations

## Files Added For This Test

- `scripts/export_small_nonzero_mamba_pose_onnx.py`
- `config/mamba_pose_onnx_safety_test.yaml`

## Output Convention

The generated ONNX model always outputs a fixed 6D vector in this order:

`[d_roll, d_pitch, d_yaw, d_tx, d_ty, d_tz]`

### Normal Mode Default

The default `normal` mode exports:

`[0.01, 0.0, 0.0, 0.01, 0.0, 0.0]`

This should be small enough to pass through the default safety limits without clamp or reject.

### Oversized Mode Default

The default `oversized` mode exports:

`[0.30, 0.0, 0.0, 0.50, 0.0, 0.0]`

With the current conservative defaults:

- `max_rotation_correction_rad = 0.10`
- `max_translation_correction_m = 0.20`

this should trigger clamp.

If you also set:

- `mamba_pose/reject_oversized_output = true`

then the same oversized output should be rejected instead of only clamped.

## Step 1: Export The Small Non-Zero Model

From the repository root:

```bash
python3 scripts/export_small_nonzero_mamba_pose_onnx.py --mode normal
```

Default path:

```text
Log/models/small_nonzero_mamba_pose.onnx
```

## Step 2: Export The Oversized Test Model

```bash
python3 scripts/export_small_nonzero_mamba_pose_onnx.py \
  --mode oversized \
  --output Log/models/oversized_mamba_pose.onnx
```

## Step 3: Prepare ROS2 Parameters

Start from:

```text
config/mamba_pose_onnx_safety_test.yaml
```

Update at least:

- `mamba_pose/model_path`

For the normal-path test, keep:

- `mamba_pose/reject_oversized_output = false`

For the reject-path test, switch to the oversized model and set:

- `mamba_pose/reject_oversized_output = true`

## Step 4: Launch FAST-LIVO2

Use your normal launch flow and add the safety-test YAML on top of your usual lidar/camera configuration.

## What To Watch In Logs

The most useful `MambaPose` fields are:

- `requested_backend`
- `active_backend`
- `model_loaded`
- `session_ready`
- `io_name_ready`
- `inference_success`
- `raw`
- `safe`
- `clamped`
- `rejected`
- `reject_reason`
- `backend_status`
- `inference_status`

## How To Judge The Result

### A. Non-Zero Output Entered The Safe Correction Path

You should see:

1. `requested_backend=onnx`
2. `active_backend=onnx`
3. `model_loaded=true`
4. `inference_success=true`
5. `raw` is non-zero
6. `safe` is also non-zero
7. `clamped=false`
8. `rejected=false`

This means decoding and safety pass-through both worked.

### B. Correction Was Clamped

Use the oversized model with:

- `mamba_pose/reject_oversized_output = false`

You should see:

1. `raw` contains the large configured values
2. `safe` is smaller than `raw`
3. `clamped=true`
4. `rejected=false`

This means the correction reached the safety layer and was clipped to the configured bounds.

### C. Correction Was Rejected

Use the oversized model with:

- `mamba_pose/reject_oversized_output = true`

You should see:

1. `raw` contains the large configured values
2. `clamped=true`
3. `rejected=true`
4. `reject_reason=oversized_output`

This means the oversized correction reached the safety layer and was explicitly refused.

## Optional Manual Override

If you want a custom fixed correction:

```bash
python3 scripts/export_small_nonzero_mamba_pose_onnx.py \
  --correction 0.02 0.00 0.00 0.03 0.00 0.00 \
  --output Log/models/custom_mamba_pose.onnx
```

## Recommended First Order

1. run `normal`
2. confirm non-zero `raw` and non-zero `safe`
3. run `oversized`
4. confirm `clamped=true`
5. enable `reject_oversized_output=true`
6. confirm `rejected=true`
