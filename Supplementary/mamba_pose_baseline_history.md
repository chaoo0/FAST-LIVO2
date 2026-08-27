# MambaPose Baseline History

This is historical engineering evidence, not the current work plan. Current
status is in `mamba_pose_project_context.md`; trusted-GT work is governed by
`mamba_pose_m3dgr_gt_quickstart.md`.

## Engineering milestones

The C++ integration established the following path:

```text
handleLIO -> StateEstimation -> PoseCompensator -> history -> ONNX Runtime
-> 6D output -> safety handling -> derived data rebuild -> UpdateVoxelMap
```

ONNX Runtime 1.18.1 was found and linked from `/opt/onnxruntime`. The MID360
launch supports MambaPose YAML and optional bag playback.

The controlled ONNX test assets verified:

| Case | Raw output | Expected result |
| --- | --- | --- |
| Normal | `[0.01, 0, 0, 0.01, 0, 0]` | accepted unchanged |
| Clamp | `[0.3, 0, 0, 0.5, 0, 0]` | safe output `[0.1, 0, 0, 0.2, 0, 0]` |
| Reject | same oversized output | clamped and rejected as `oversized_output` |

These cases are regression/safety evidence only. Their YAML files must never
be used to export formal training data.

## Stationary static-zero baseline

The stationary CSV pipeline was implemented to validate data flow, training,
and ONNX interface. It produces basic18 windows:

```text
pos(3), quaternion(4), velocity(3), gyro bias(3), accel bias(3),
effective_feature_num, avg_residual
```

The static label was six zeros in legacy Euler/translation ordering. A small
FlattenMLP trained and exported ONNX successfully; an additional raw-input
wrapper embeds saved normalization because the C++ path does not normalize its
basic18 tensors.

This was a no-op engineering baseline. Its nonzero residual predictions make
it unsuitable for state writeback; it remains observe-only.

## Dynamic pseudo-smooth baseline

The dynamic MID360 run `mid360_fastlivo_mamba_20260605_161447` produced:

```text
5127 raw rows -> 5117 clean rows -> 5108 T=10 windows
```

`build_mamba_pose_pseudo_label_dataset.py` generated labels relative to a
same-segment smoothed trajectory. The selected 0.5-second dataset trained a
FlattenMLP and exported:

```text
models/pseudo_smooth_mamba_pose.pt
models/pseudo_smooth_mamba_pose.onnx
models/pseudo_smooth_mamba_pose_raw_input.onnx
```

The raw-input model is valid for runtime inference diagnostics with
`apply_correction_en=false`; the normalized-input model is offline-only.
Pseudo-smooth labels are not Mocap ground truth and do not establish dynamic
compensation accuracy.

## Why this history is superseded

Both baselines validate the plumbing and provide comparison points, but neither
defines the final target. M3DGR uses a right SE(3) target in local IMU
coordinates, while the legacy writeback applies Euler/world translation.
Until M3DGR trust gates and a runtime-contract migration are complete, keep all
new GT models in shadow or observe-only mode.
