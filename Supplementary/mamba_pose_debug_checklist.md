# Mamba Pose Debug Checklist

This checklist is for the current ONNX-based pose compensation validation flow in FAST-LIVO2.

Current main launch:

```text
launch/mapping_mid360.launch.py
```

The same ONNX / safety capability that was first validated in the Avia flow has now been synchronized to the MID360 launch flow, including automatic rosbag playback support.

---

## Current Launch Arguments

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

Meaning:

- `play_bag`: auto-run `ros2 bag play`
- `bag_path`: bag to play
- `bag_loop`: add `--loop`
- `bag_clock`: add `--clock`

---

## Pre-Flight Checklist

Before running any ONNX validation round, confirm all of the following:

1. FAST-LIVO2 was rebuilt with ONNX Runtime enabled.
2. `/opt/onnxruntime` is available and linked correctly.
3. `mamba_pose/backend_type` is set to `onnx`.
4. `mamba_pose/onnx_input_name` matches the model input name.
5. `mamba_pose/onnx_output_name` matches the model output name.
6. `mamba_pose/debug_log_en` is `true`.
7. `mamba_pose/enabled` is `true`.
8. `mamba_pose/apply_correction_en` matches the intended runtime mode.
9. `mamba_pose/model_path` is an absolute path and points to the intended test model.
10. `mamba_pose/history_len=10` and `mamba_pose/min_ready_frames=10` unless intentionally changed.
11. Safety limits are known before the test:
    `max_rotation_correction_rad = 0.10`
    `max_translation_correction_m = 0.20`

---

## Test YAML Usage Rules

These files are test-only:

- `config/mamba_pose_onnx_normal_test.yaml`
- `config/mamba_pose_onnx_safety_test.yaml`
- `config/mamba_pose_onnx_reject_test.yaml`

Do not use those files for formal training-data export.

Formal export must use:

```text
config/mamba_pose_train_export_dummy.yaml
```

Reason:

> The ONNX test models output fixed corrections. Using them during real CSV export would contaminate the original FAST-LIVO2 state history.

---

## Recommended Test Order

Run the three rounds in this order:

1. normal non-zero output test
2. oversized output with clamp test
3. oversized output with reject test

This order verifies:

- normal ONNX inference first
- then clamp behavior
- then explicit rejection behavior

---

## Models Used In Each Round

- Round 1 uses `Log/models/small_nonzero_mamba_pose.onnx`
- Round 2 uses `Log/models/oversized_mamba_pose.onnx`
- Round 3 also uses `Log/models/oversized_mamba_pose.onnx`, but with reject enabled

---

## Key Log Fields To Watch

Focus on these `MambaPose` fields:

- `applied`
- `apply_correction_en`
- `requested_backend`
- `active_backend`
- `model_loaded`
- `session_ready`
- `io_name_ready`
- `inference_success`
- `fallback`
- `backend_status`
- `backend_error`
- `inference_status`
- `seq_len`
- `feature_dim`
- `flat_input_len`
- `output_dim`
- `raw`
- `safe`
- `clamped`
- `rejected`
- `reject_reason`

---

## Three-Round Comparison Table

| Test Name | YAML | Model | `reject_oversized_output` | Expected `raw` | Expected `safe` | Expected `clamped` | Expected `rejected` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Normal non-zero output | `config/mamba_pose_onnx_normal_test.yaml` | `small_nonzero_mamba_pose.onnx` | `false` | close to `[0.01, 0, 0, 0.01, 0, 0]` | same as `raw` | `false` | `false` |
| Oversized output + clamp | `config/mamba_pose_onnx_safety_test.yaml` | `oversized_mamba_pose.onnx` | `false` | close to `[0.3, 0, 0, 0.5, 0, 0]` | clipped to bounds | `true` | `false` |
| Oversized output + reject | `config/mamba_pose_onnx_reject_test.yaml` | `oversized_mamba_pose.onnx` | `true` | close to `[0.3, 0, 0, 0.5, 0, 0]` | rejected candidate | `true` | `true` |

---

## Verified MID360 Status

The MID360 launch flow has already passed the three expected behaviors:

### Normal Path

Verified logs include:

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

### Clamp Path

Expected and verified:

```text
raw=[0.3, 0, 0, 0.5, 0, 0]
safe=[0.1, 0, 0, 0.2, 0, 0]
clamped=true
rejected=false
reject_reason=none
```

### Reject Path

Expected and verified:

```text
raw=[0.3, 0, 0, 0.5, 0, 0]
safe=[0.1, 0, 0, 0.2, 0, 0]
clamped=true
rejected=true
reject_reason=oversized_output
```

---

## Static Zero Runtime Verification

The current offline training loop can now export:

```text
Log/models/static_zero_mamba_pose_raw_input.onnx
```

This model is only a stationary no-op baseline. It should be used to verify the ONNX runtime loop and FAST-LIVO2 integration, not real dynamic compensation ability.

Do not directly wire:

```text
Log/models/static_zero_mamba_pose.onnx
```

into the current FAST-LIVO2 backend, because that older export expects already-normalized input.

Current safe runtime starting point:

```text
config/mamba_pose_static_zero_runtime_observe_only.yaml
```

This observe-only configuration keeps `mamba_pose/enabled=true` so history, ONNX inference, and logs still run, but sets:

```text
mamba_pose/apply_correction_en=false
```

so the compensated state is not written back into `_state` or `voxelmap_manager->state_`.

### Runtime Setup To Check

For the runtime no-op verification round, confirm:

- `mamba_pose/backend_type=onnx`
- `mamba_pose/model_path` points to `Log/models/static_zero_mamba_pose_raw_input.onnx`
- `mamba_pose/onnx_input_name=input`
- `mamba_pose/onnx_output_name=output`
- `mamba_pose/history_len=10`
- `mamba_pose/min_ready_frames=10`
- `mamba_pose/apply_correction_en=false` for the first observe-only validation round

### Success Signals To Watch

At runtime, confirm:

1. `active_backend=onnx`
2. `model_loaded=true`
3. `session_ready=true`
4. `io_name_ready=true`
5. `inference_success=true`
6. `output_dim=6`
7. `apply_correction_en=false`
8. `applied=false`
9. `raw` and `safe` are visible in logs for observation
10. FAST-LIVO2 state and map do not jump, because correction writeback is disabled

Expected interpretation:

- ONNX runtime path is alive
- export wrapper shape `[10, 18] -> [6]` is compatible
- observe-only mode is preventing `_state` corruption even if the current no-op baseline later drifts

Important note:

> For the current runtime round, `raw` / `safe` being exactly near zero is no longer the only pass criterion. If the model output drifts over time, the main safety success condition is still `applied=false` together with stable FAST-LIVO2 state, because this round is meant to observe model behavior without closing the loop.

### Failure Focus For This Round

If the no-op baseline fails at runtime, check these first:

| Symptom | First Check |
| --- | --- |
| `active_backend` is not `onnx` | `backend_type`, build state, ONNX Runtime availability |
| `model_loaded=false` | `mamba_pose/model_path` points to `Log/models/static_zero_mamba_pose_raw_input.onnx` |
| `io_name_ready=false` | `onnx_input_name=input`, `onnx_output_name=output` |
| `inference_success=false` | ONNX export shape, runtime tensor shape, model compatibility |
| `apply_correction_en=true` during observe-only test | wrong YAML or parameter override |
| `applied=true` during observe-only test | parameter load failure or logic regression in `LIVMapper` |
| `raw` grows large over time but map stays stable | observe-only mode is working; this confirms the model is not ready for closed-loop apply |
| `raw` grows large and map also jumps | correction writeback is still enabled somewhere in runtime config |
| `safe` is not close to zero while `raw` is close to zero | safety thresholds or later runtime-side handling needs inspection |

---

## Success Conditions By Round

### Round 1 Success

1. `requested_backend=onnx`
2. `active_backend=onnx`
3. `model_loaded=true`
4. `session_ready=true`
5. `io_name_ready=true`
6. `inference_success=true`
7. `output_dim=6`
8. `raw` is non-zero
9. `safe` matches `raw`
10. `clamped=false`
11. `rejected=false`

### Round 2 Success

1. `requested_backend=onnx`
2. `active_backend=onnx`
3. `inference_success=true`
4. `raw` is oversized and non-zero
5. `safe` is smaller than `raw`
6. `clamped=true`
7. `rejected=false`

### Round 3 Success

1. `requested_backend=onnx`
2. `active_backend=onnx`
3. `inference_success=true`
4. `raw` is oversized and non-zero
5. `clamped=true`
6. `rejected=true`
7. `reject_reason=oversized_output`

---

## Failure Diagnosis Table

| Symptom | Likely Layer | What It Usually Means | First Thing To Check |
| --- | --- | --- | --- |
| `active_backend` is not `onnx` | backend selection / build | runtime fell back or ONNX support was not active | rebuild state, `backend_type`, ONNX Runtime availability |
| `model_loaded=false` | model loading | wrong path or missing file | `mamba_pose/model_path` |
| `session_ready=false` | session creation | ONNX Runtime could not create a session | model validity, ORT compatibility |
| `io_name_ready=false` | IO binding | configured names do not match the model | `onnx_input_name`, `onnx_output_name` |
| `inference_success=false` | inference execution | shape or runtime execution mismatch | `seq_len`, `feature_dim`, `backend_status`, `inference_status` |
| `output_dim != 6` | output decode | model output shape is wrong | model export contract |
| `safe` is unexpected but `raw` looks correct | safety layer | clamp / reject policy is responsible, not the model | thresholds, `reject_oversized_output` |

---

## Recommended Command Pattern

MID360 normal test:

```bash
ros2 launch fast_livo mapping_mid360.launch.py \
  mamba_pose_params_file:=/home/liu/fast_livo2/src/FAST-LIVO2/config/mamba_pose_onnx_normal_test.yaml \
  play_bag:=true \
  bag_path:=/home/liu/rosbags/mid360_fastlivo_mamba_20260519_211645 \
  bag_loop:=false \
  bag_clock:=false
```

Watch logs:

```bash
grep -E "MambaPose|requested_backend|active_backend|model_loaded|session_ready|io_name_ready|inference_success|backend_status|backend_error|inference_status|clamped|rejected|reject_reason|raw=|safe="
```
