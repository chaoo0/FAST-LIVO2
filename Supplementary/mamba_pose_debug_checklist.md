# MambaPose Runtime Debug Checklist

Use this document only to diagnose MID360 runtime behavior. Current project
state and the M3DGR workflow are intentionally documented elsewhere.

## Before launch

1. Build/source the `fast_livo` workspace and expose `/opt/onnxruntime/lib` in
   `LD_LIBRARY_PATH` when using ONNX.
2. Select the YAML by purpose:

   | Purpose | YAML |
   | --- | --- |
   | Formal MID360 export | `config/mamba_pose_train_export_20260605_161447.yaml` |
   | Runtime safety regression | `config/mamba_pose_onnx_*_test.yaml` |
   | Static-zero observe-only | `config/mamba_pose_static_zero_runtime_observe_only.yaml` |
   | Pseudo-smooth observe-only | `config/mamba_pose_pseudo_smooth_runtime_observe_only_20260605_161447.yaml` |

3. For any learned model, confirm its path ends in `_raw_input.onnx`.
4. For every baseline or new GT model, confirm
   `mamba_pose/apply_correction_en=false`.
5. Ensure formal exports use a dummy backend. Never export using a fixed-output
   safety-test YAML.

## Required logs

After the history warms up, inspect:

```text
requested_backend, active_backend, model_loaded, session_ready, io_name_ready
inference_success, fallback, backend_error, seq_len, feature_dim, output_dim
raw, safe, clamped, rejected, reject_reason, apply_correction_en, applied
```

Expected learned-model diagnostics: `seq_len=10`, `feature_dim=18`,
`output_dim=6`, `inference_success=true`, `fallback=false`, and `applied=false`.

## Diagnosis table

| Signal | Likely cause | Check |
| --- | --- | --- |
| `model_loaded=false` | wrong/missing model | absolute raw-input ONNX path |
| `session_ready=false` | ONNX Runtime setup | installed library and `LD_LIBRARY_PATH` |
| `io_name_ready=false` | ONNX contract mismatch | names `input`/`output`, shapes `[10,18]` and `[6]` |
| `fallback=true` | backend failed or is dummy | `backend_error` and selected YAML |
| no inference | history not ready | `history_size`, `ready_flag`, `min_ready_frames` |
| `applied=true` unexpectedly | unsafe parameter override | effective `apply_correction_en` value |
| safety rejection | output violates limits | `raw`, `safe`, `clamped`, `reject_reason` |

## Current dynamic observe-only command

```bash
cd /home/liu/fast_livo2
source /opt/ros/humble/setup.bash
source /home/liu/fast_livo2/install/setup.bash
export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH

ros2 launch fast_livo mapping_mid360.launch.py \
  use_rviz:=False play_bag:=True \
  bag_path:=/home/liu/data_sda5/rosbags/mid360_fastlivo_mamba_20260605_161447 \
  bag_loop:=False bag_clock:=False \
  mamba_pose_params_file:=/home/liu/fast_livo2/src/FAST-LIVO2/config/mamba_pose_pseudo_smooth_runtime_observe_only_20260605_161447.yaml
```

This validates inference only. It is not evidence that pseudo-label corrections
are safe or accurate enough for closed-loop writeback.
