# MambaPose ONNX Safety Regression

This is the authoritative procedure for the already-verified fixed-output
normal, clamp, and reject checks. It does not validate a learned model.

## Expected behavior

| YAML | Fixed raw output | Expected safety result |
| --- | --- | --- |
| `mamba_pose_onnx_normal_test.yaml` | `[0.01,0,0,0.01,0,0]` | accepted unchanged |
| `mamba_pose_onnx_safety_test.yaml` | `[0.3,0,0,0.5,0,0]` | clamped to `[0.1,0,0,0.2,0,0]` |
| `mamba_pose_onnx_reject_test.yaml` | `[0.3,0,0,0.5,0,0]` | clamped, then rejected as `oversized_output` |

## Run a case

Replace `<yaml>` with exactly one file from the table.

```bash
cd /home/liu/fast_livo2
source /opt/ros/humble/setup.bash
source /home/liu/fast_livo2/install/setup.bash
export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH

ros2 launch fast_livo mapping_mid360.launch.py \
  mamba_pose_params_file:=/home/liu/fast_livo2/src/FAST-LIVO2/config/<yaml> \
  play_bag:=true \
  bag_path:=/home/liu/data_sda5/rosbags/mid360_fastlivo_mamba_20260605_161447 \
  bag_loop:=false bag_clock:=false
```

## Pass criteria

Require all backend fields to report ready/success, `seq_len=10`,
`feature_dim=18`, `output_dim=6`, no fallback, and the `raw`, `safe`,
`clamped`, `rejected`, and `reject_reason` values from the table.

These test YAMLs may apply their controlled correction by design; they are
isolated safety regressions. Never use them for training-data export or as a
learned-model acceptance test.
