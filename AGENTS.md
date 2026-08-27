# FAST-LIVO2 + MambaPose Agent Guide

## Scope and mandatory reading

This repository is the FAST-LIVO2 + MambaPose research prototype. Work only on
Innovation Point 1: temporal pose compensation. Do not restart FAST-LIVO2 or
expand to later innovation points unless explicitly requested.

Before any task, read [mamba_pose_project_context.md](mamba_pose_project_context.md).
For M3DGR work, also read
[Supplementary/mamba_pose_m3dgr_gt_quickstart.md](Supplementary/mamba_pose_m3dgr_gt_quickstart.md).

## Current state and next action

The MID360 ONNX chain and normal/clamp/reject safety tests are verified.
`static_zero` and `pseudo_smooth_reference` are engineering baselines only;
both must remain observe-only.

The active work is M3DGR indoor Mocap GT. Acquire and SHA256-verify exactly
the frozen `Varying-illu02` Pilot bag, convert/audit it, then run the M3DGR
LIO export. Its alignment must remain `alignment_candidate` until the physical
Mocap rigid-body frame is externally confirmed. Do not start trusted-GT
training or GT-model state writeback before that gate passes.

## Runtime and data invariants

- `mamba_pose/enabled` controls buffering and inference; `apply_correction_en`
  controls state writeback. The default and validation mode is `false`.
- FAST-LIVO2 sends raw basic18 features to ONNX. Runtime models must be the
  `*_raw_input.onnx` variants.
- Formal M3DGR export uses dummy backend, `apply_correction_en=false`, frozen
  splits in `config/m3dgr_dataset_manifest.yaml`, and train-only normalization.
- M3DGR labels are right perturbations: `Log_SE3(inv(T_est) * T_ref)` in
  `[delta_theta, delta_rho]`. Do not filter valid labels with runtime limits.
- The legacy C++ apply path uses Euler/world-translation semantics and is not
  compatible with these labels; right-SE(3) models remain observe-only.

## Repository and commands

- Package: `fast_livo`; workspace: `/home/liu/fast_livo2`; ROS 2 Humble.
- Main MID360 launch: `launch/mapping_mid360.launch.py`.
- M3DGR launch: `launch/mapping_m3dgr_lio.launch.py`.
- Latest MID360 artifact root:
  `Log/runs/mid360_fastlivo_mamba_20260605_161447`.
- Build C++ changes with:

  ```bash
  cd /home/liu/fast_livo2
  source /opt/ros/humble/setup.bash
  colcon build --symlink-install --packages-select fast_livo \
    --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DENABLE_ONNXRUNTIME=ON -DONNXRUNTIME_ROOT=/opt/onnxruntime
  ```

## Change constraints

Do not modify without an explicit request: `src/voxel_map.cpp`,
`include/voxel_map.h`, `src/vio.cpp`, or `include/vio.h`.

Avoid modifying `src/pose_compensator.cpp`, `include/pose_compensator.h`,
`src/LIVMapper.cpp`, `include/LIVMapper.h`, `CMakeLists.txt`, and
`launch/mapping_avia.launch.py` unless the task requires it.

After code or script changes, sync the relevant Markdown document and report
files changed, core-SLAM impact, build command, runtime/test command, and
expected output.
