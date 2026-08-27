# FAST-LIVO2 + MambaPose: Current State Snapshot

Updated: 2026-08-27. This is the sole current-state document. Historical
baseline details live in
[Supplementary/mamba_pose_baseline_history.md](Supplementary/mamba_pose_baseline_history.md);
procedure details live in the relevant quickstart/checklist.

## Goal and boundaries

The project validates and eventually trains a temporal MambaPose 6DoF
compensator inside FAST-LIVO2:

```text
LIO state -> PoseCompensator -> basic18 history -> ONNX -> 6D correction
          -> safety gate -> derived-state rebuild -> voxel-map update
```

Only Innovation Point 1 is in scope. Do not redesign FAST-LIVO2 or add later
map/loop-closure innovation points.

## Verified engineering baseline

- ROS 2 Humble package `fast_livo` builds with ONNX Runtime at
  `/opt/onnxruntime`.
- The MID360 path loads YAML, buffers history, runs ONNX inference, applies the
  safety layer, rebuilds derived state, and reaches map update.
- MID360 normal, clamp, and reject safety cases are verified.
- Data export and offline tooling are implemented: CSV clean/interpolate,
  T=10 sequence construction, basic18 normalization, static-zero and
  pseudo-smooth baseline data/train/export paths.
- Runtime defaults to observe-only: `mamba_pose/apply_correction_en=false`.
  Raw-input ONNX wrappers are required because C++ supplies raw basic18 input.

The existing `static_zero` and `pseudo_smooth_reference` models prove the
engineering interface only. They are not trusted dynamic supervision and must
not be used for closed-loop correction.

## Latest usable MID360 artifact

The current dynamic reference run is:

```text
rosbag: /home/liu/data_sda5/rosbags/mid360_fastlivo_mamba_20260605_161447
artifacts: Log/runs/mid360_fastlivo_mamba_20260605_161447
```

Verified results:

```text
raw CSV rows: 5127
clean rows: 5117
T=10 windows: 5108
segments: 1
raw basic18 non-finite values: 0 after cleaning
```

The run directory contains the cleaned/interpolated CSVs, sequence and
normalization NPZs, 0.5/1.0/2.0-second pseudo-label variants, and the selected
0.5-second FlattenMLP checkpoint plus normalized/raw-input ONNX exports.

Use run-specific paths in new commands. Do not use test YAMLs for formal CSV
export; use `config/mamba_pose_train_export_20260605_161447.yaml` for the
current MID360 run or `config/mamba_pose_train_export_dummy.yaml` as the
generic export template.

## M3DGR trusted-GT status

M3DGR is now the intended supervision source, using MID360 + built-in IMU;
D435i is deferred. Implemented, regression-tested components include:

- frozen split/checksum manifest and fail-closed artifact paths;
- ROS1 verification, ROS2 conversion, topic/point-time audit, and optional
  separate Livox-driver2 compatibility rewrite;
- M3DGR LIO launch with image disabled, dummy backend, per-run export, and
  `current`/`kalibr_variance` covariance profiles;
- TUM audit, timestamp offset search, interpolation, world alignment,
  hand-eye consistency estimation, and provenance reports;
- right-SE(3) dataset assembly with formal trust gate, bag isolation,
  train-only normalization, and large-label preservation.

On 2026-08-08, the package/launch smoke checks and all six synthetic M3DGR
regression tests passed. The public `Varying-illu02.txt` GT passed SHA256,
strict timestamp, and quaternion checks (43,550 poses over 146.49 s).

No real M3DGR Pilot bag, converted bag, LIO CSV, alignment report, or formal
dataset exists locally. The physical definition of the Mocap rigid body is
also unconfirmed. Therefore no artifact is eligible as `trusted_gt`.

## Current single next task

1. Review the external terms and obtain exactly the manifest-frozen
   `Varying-illu02` bag.
2. SHA256-verify, convert, and audit it using the M3DGR quickstart.
3. Run `mapping_m3dgr_lio.launch.py` with `covariance_profile:=current`, dummy
   backend, and observe-only output to create the first real LIO CSV.
4. Audit/align it against Mocap GT, retaining only `alignment_candidate` until
   external confirmation of the rigid-body physical frame.

Do not train trusted GT, freeze a covariance profile, or modify state-writeback
semantics before the gates in the M3DGR quickstart have passed.

## Critical contracts

- M3DGR target: `xi = Log_SE3(inv(T_est) * T_ref)` ordered as
  `[delta_theta, delta_rho]`.
- Valid large training errors remain in the dataset; runtime safety limits do
  not filter them.
- Formal assembly requires `trusted_gt`, one confirmed frame convention and
  extrinsic, stable offsets, frozen split membership, and train-only feature
  statistics.
- Current `applyCorrection()` uses legacy Euler plus world-frame translation;
  it is not equivalent to right-SE(3). GT ONNX models must stay observe-only
  until that contract is deliberately migrated and tested.

## Authority map

| Need | Authoritative source |
| --- | --- |
| Agent constraints and immediate next action | `AGENTS.md` |
| Current state and artifact baseline | This file |
| M3DGR acquisition through formal assembly | `Supplementary/mamba_pose_m3dgr_gt_quickstart.md` |
| MID360 runtime diagnostics | `Supplementary/mamba_pose_debug_checklist.md` |
| ONNX input/output interface | `Supplementary/mamba_pose_onnx_quickstart.md` |
| Safety test procedure | `Supplementary/mamba_pose_safety_test.md` |
| Baseline experiment history | `Supplementary/mamba_pose_baseline_history.md` |

## Key paths

```text
launch/mapping_mid360.launch.py
launch/mapping_m3dgr_lio.launch.py
config/m3dgr_dataset_manifest.yaml
config/m3dgr_mid360_lio.yaml
config/mamba_pose_m3dgr_train_export_dummy.yaml
scripts/prepare_m3dgr_bag.py
scripts/align_m3dgr_mocap_gt.py
scripts/build_m3dgr_mocap_dataset.py
tests/test_m3dgr_pipeline.py
```
