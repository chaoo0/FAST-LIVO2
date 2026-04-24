# Mamba Pose Debug Checklist

This checklist is for the first local safety validation of the ONNX-based pose compensation path in FAST-LIVO2.

It assumes you already have:

- a build of FAST-LIVO2 with optional ONNX Runtime support available
- the debug logs enabled through `mamba_pose/debug_log_en: true`
- the test asset scripts already added in this repository

## Pre-Flight Checklist

Before running any of the three tests, confirm all of the following:

1. FAST-LIVO2 was rebuilt after enabling ONNX Runtime in CMake.
2. The ONNX model file exists on disk.
3. `mamba_pose/backend_type` is set to `onnx`.
4. `mamba_pose/onnx_input_name` matches the model input name.
5. `mamba_pose/onnx_output_name` matches the model output name.
6. `mamba_pose/debug_log_en` is `true`.
7. `mamba_pose/enabled` is `true`.
8. `mamba_pose/model_path` is an absolute path and points to the intended test model.
9. `mamba_pose/history_len` and `mamba_pose/min_ready_frames` are reasonable, so the model can become ready quickly.
10. Safety limits are known before the test:
    `max_rotation_correction_rad = 0.10`
    `max_translation_correction_m = 0.20`

## Recommended Test Order

Run the three tests in this order:

1. normal non-zero output test
2. oversized output with clamp test
3. oversized output with reject test

This order matters because it first verifies the normal pass-through path, then verifies clamp, and only after that verifies explicit rejection.

## Models Used In Each Round

- Round 1 uses the `normal` model exported by:
  `scripts/export_small_nonzero_mamba_pose_onnx.py --mode normal`
- Round 2 uses the `oversized` model exported by:
  `scripts/export_small_nonzero_mamba_pose_onnx.py --mode oversized`
- Round 3 also uses the `oversized` model, but changes the reject policy in YAML

## Key Log Fields To Watch

For all three rounds, focus on these `MambaPose` log fields:

- `requested_backend`
- `active_backend`
- `model_loaded`
- `session_ready`
- `io_name_ready`
- `inference_success`
- `output_dim`
- `raw`
- `safe`
- `clamped`
- `rejected`
- `reject_reason`
- `backend_status`
- `inference_status`

## Three-Round Comparison Table

| Test Name | Model Type | `reject_oversized_output` | Expected `requested_backend` | Expected `active_backend` | Expected `inference_success` | Expected `raw correction` | Expected `safe correction` | Expected `clamped` | Expected `rejected` | Expected `reject_reason` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Normal non-zero output | normal | `false` | `onnx` | `onnx` | `true` | non-zero and close to `[0.01, 0.0, 0.0, 0.01, 0.0, 0.0]` | non-zero and equal or very close to `raw` | `false` | `false` | `none` |
| Oversized output + clamp | oversized | `false` | `onnx` | `onnx` | `true` | non-zero and close to `[0.30, 0.0, 0.0, 0.50, 0.0, 0.0]` | non-zero but smaller than `raw`, clipped by safety limits | `true` | `false` | `none` |
| Oversized output + reject | oversized | `true` | `onnx` | `onnx` | `true` | non-zero and close to `[0.30, 0.0, 0.0, 0.50, 0.0, 0.0]` | clamped candidate may still be shown, but final application should be rejected | `true` | `true` | `oversized_output` |

## Per-Round YAML Focus

### Round 1: Normal Non-Zero Output

Use:

- `config/mamba_pose_onnx_normal_test.yaml`

Confirm:

- `mamba_pose/model_path` points to the normal model
- `mamba_pose/reject_oversized_output: false`

### Round 2: Oversized Output + Clamp

Use:

- `config/mamba_pose_onnx_safety_test.yaml`
  or keep the same file and only change `mamba_pose/model_path`

Confirm:

- `mamba_pose/model_path` points to the oversized model
- `mamba_pose/reject_oversized_output: false`

### Round 3: Oversized Output + Reject

Use:

- `config/mamba_pose_onnx_reject_test.yaml`

Confirm:

- `mamba_pose/model_path` points to the oversized model
- `mamba_pose/reject_oversized_output: true`

## Success Conditions By Round

### Round 1 Success

All of the following should hold:

1. `requested_backend=onnx`
2. `active_backend=onnx`
3. `model_loaded=true`
4. `session_ready=true`
5. `io_name_ready=true`
6. `inference_success=true`
7. `output_dim=6`
8. `raw` is non-zero
9. `safe` is non-zero
10. `clamped=false`
11. `rejected=false`

### Round 2 Success

All of the following should hold:

1. `requested_backend=onnx`
2. `active_backend=onnx`
3. `inference_success=true`
4. `raw` is oversized and non-zero
5. `safe` is smaller than `raw`
6. `clamped=true`
7. `rejected=false`

### Round 3 Success

All of the following should hold:

1. `requested_backend=onnx`
2. `active_backend=onnx`
3. `inference_success=true`
4. `raw` is oversized and non-zero
5. `clamped=true`
6. `rejected=true`
7. `reject_reason=oversized_output`

## Failure Diagnosis Table

| Symptom | Likely Layer | What It Usually Means | First Thing To Check |
| --- | --- | --- | --- |
| `active_backend` is not `onnx` | backend selection / build | ONNX backend was not actually active, or runtime fell back | `backend_type`, ONNX Runtime availability, rebuild logs |
| `model_loaded=false` | model loading | bad path or model file missing | `mamba_pose/model_path`, file existence |
| `session_ready=false` | session creation | ONNX Runtime could not build a session from this model | model validity, ONNX opset, ORT compatibility |
| `io_name_ready=false` | model IO binding | configured names do not match model names | `onnx_input_name`, `onnx_output_name` |
| `inference_success=false` | inference execution | tensor creation, shape check, or runtime execution failed | `inference_status`, `backend_status`, model input/output contract |
| `output_dim != 6` | output decode | model output length does not match the expected 6D correction | model output shape |
| `raw` is zero but it should be non-zero | model or wrong asset | wrong model file, wrong export mode, or output path mismatch | the model path and export command used |
| `safe` is zero while `raw` is non-zero | safety layer | output may have been rejected or collapsed by protection logic | `clamped`, `rejected`, `reject_reason`, thresholds |
| `rejected=true` when this round should not reject | safety policy | thresholds too tight or wrong reject policy for this round | `reject_oversized_output`, configured max rotation/translation |

## Practical Notes

- Wait until the history becomes ready before judging inference behavior.
- Always compare `raw` and `safe` together.
- If `raw` is correct but `safe` is unexpected, the issue is usually in the safety configuration, not the ONNX model itself.
- If `active_backend=onnx` but `inference_success=false`, the issue is typically in IO names, shape, or runtime compatibility.

## Recommended Command Flow

1. export the normal model
2. run the normal test YAML
3. confirm pass-through behavior
4. export the oversized model
5. run the clamp configuration
6. confirm clipped safe correction
7. run the reject configuration
8. confirm explicit rejection
