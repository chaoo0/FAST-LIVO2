# M3DGR Mocap GT Quickstart

## Purpose And Current Gate

This workflow replaces `pseudo_smooth_reference` with a full 6DoF Mocap
reference for Innovation Point 1. It starts with MID360 + built-in IMU LIO;
D435i is intentionally deferred.

The current physical frame of the published Mocap rigid body is not documented
in the M3DGR README, calibration page, or paper. Therefore:

- numerical alignment alone does not make a trusted dataset;
- unconfirmed runs must remain `alignment_candidate`;
- formal dataset assembly accepts only `trusted_gt` reports;
- hand-eye consistency does not replace confirmation of the physical rigid-body definition.

## Frozen Split

- train: `Dynamic01`, `Occlusion01`, `Varying-illu01`, `Dark03`, `Wheel-float01`, `Sha-turn01`
- val: `Dynamic02`, `Wheel-float02`
- test: `Occlusion02`, `Varying-illu02`, `Dark04`, `Sha-turn02`
- Pilot: `Varying-illu02`

The split and official bag SHA256 values are frozen in:

```text
config/m3dgr_dataset_manifest.yaml
```

Empty paths mean missing data. Do not replace a missing sequence with another bag.

## 1. Download And Verify

First review the external data-page terms. The repository is MIT licensed, but
the externally hosted bag files do not currently expose a separate license in
the repository.

The Pilot GT can be downloaded and promoted only after checksum validation:

```bash
python3 scripts/prepare_m3dgr_bag.py download \
  --manifest config/m3dgr_dataset_manifest.yaml \
  --sequence Varying-illu02 \
  --kind gt \
  --output /path/to/M3DGR/Varying-illu02.txt \
  --accept-external-data-terms
```

For a downloaded ROS1 bag:

```bash
python3 scripts/prepare_m3dgr_bag.py verify \
  --manifest config/m3dgr_dataset_manifest.yaml \
  --sequence Varying-illu02 \
  --bag /path/to/M3DGR/Varying-illu02.bag
```

Do not use a bag if its SHA256 differs from the manifest.

## 2. Convert And Audit The Bag

Use the official `rosbags-convert` path through the checked wrapper:

```bash
python3 scripts/prepare_m3dgr_bag.py convert \
  --manifest config/m3dgr_dataset_manifest.yaml \
  --sequence Varying-illu02 \
  --src /path/to/M3DGR/Varying-illu02.bag \
  --dst /path/to/M3DGR/Varying-illu02_ros2
```

This invokes `rosbags-convert --dst-version 8`. Then inspect topics, rates,
message counts, and sampled per-point time offsets:

```bash
python3 scripts/prepare_m3dgr_bag.py inspect \
  --bag /path/to/M3DGR/Varying-illu02_ros2 \
  --report-json /path/to/M3DGR/Varying-illu02_ros2_audit.json
```

Expected topics:

```text
/livox/mid360/lidar
/livox/mid360/imu
```

If `driver_compatibility=direct_fast_livo2`, use the bag directly. If it is
`requires_offline_driver2_rewrite`, create a separate LIO-only driver2 bag:

```bash
python3 scripts/rewrite_m3dgr_livox_driver2_bag.py \
  --input-bag /path/to/M3DGR/Varying-illu02_ros2 \
  --output-bag /path/to/M3DGR/Varying-illu02_ros2_driver2 \
  --report-json /path/to/M3DGR/Varying-illu02_driver2_rewrite.json
```

The rewrite copies serialized point payloads and timestamps verbatim and keeps
only MID360 lidar + IMU. Livox points need not be sorted by `offset_time`;
nonzero offsets and matching point counts are the preservation criteria.

## 3. Run FAST-LIVO2 LIO Export

Build and source the package, then run:

```bash
ros2 launch fast_livo mapping_m3dgr_lio.launch.py \
  sequence:=Varying-illu02 \
  bag_path:=/path/to/M3DGR/Varying-illu02_ros2_driver2 \
  covariance_profile:=current \
  play_bag:=true \
  bag_clock:=true \
  use_rviz:=false
```

The export is written to:

```text
Log/runs/m3dgr_Varying-illu02/mamba_pose_train_data.csv
```

The launch enforces:

```text
img_en=0
lidar=/livox/mid360/lidar
imu=/livox/mid360/imu
mid360 -> mid360_imu T=[-0.011,-0.02329,0.04412], R=I
backend_type=dummy
apply_correction_en=false
```

It refuses to append to an existing non-empty CSV unless
`allow_append:=true` is explicitly supplied.

For training-only covariance comparison, rerun training sequences with:

```text
covariance_profile:=kalibr_variance
```

This uses `gyr_n^2` and `acc_n^2`; it is a candidate, not an assumed correct
mapping of Kalibr density into the existing FAST-LIVO2 covariance semantics.
Use `scripts/evaluate_m3dgr_lio_profiles.py` on all six training bags and freeze
the selected profile. Validation and test bags must not participate.

## 4. Audit Mocap And Align

Clean the LIO export with the existing X-side filter first:

```bash
python3 scripts/analyze_mamba_pose_train_data.py \
  --input Log/runs/m3dgr_Varying-illu02/mamba_pose_train_data.csv \
  --output-clean Log/runs/m3dgr_Varying-illu02/mamba_pose_train_data_clean.csv \
  --output-report Log/runs/m3dgr_Varying-illu02/mamba_pose_train_data_report.txt \
  --save-clean
```

Audit the TUM file independently:

```bash
python3 scripts/audit_m3dgr_mocap_tum.py \
  --input /path/to/M3DGR/Varying-illu02.txt \
  --expected-sha256 293f0ef2b9c552f14772e512a9f6634d84cb09224bb3ddc50f2c6f6c04535d0f \
  --output-json Log/runs/m3dgr_Varying-illu02/mocap_audit.json
```

Until the rigid-body definition is confirmed, only create a candidate:

```bash
python3 scripts/align_m3dgr_mocap_gt.py \
  --lio-csv Log/runs/m3dgr_Varying-illu02/mamba_pose_train_data_clean.csv \
  --mocap-tum /path/to/M3DGR/Varying-illu02.txt \
  --output-csv Log/runs/m3dgr_Varying-illu02/mamba_pose_train_data_aligned.csv \
  --report-json Log/runs/m3dgr_Varying-illu02/mocap_alignment.json \
  --run-id Varying-illu02 \
  --candidate-identity-extrinsic \
  --rigid-body-frame-status unconfirmed \
  --extrinsic-source pipeline_debug_only
```

This performs:

1. strict timestamp and quaternion validation;
2. fixed-offset search over +/-0.2 s with 1 ms steps using speed correlations;
3. position interpolation + quaternion SLERP with a 20 ms bracket limit;
4. first-2-second world alignment only;
5. GT field/provenance filling plus JSON sidecar reporting.

For multi-run hand-eye estimation, prepare a YAML with at least two
`run_id/lio_csv/mocap_tum/time_offset_sec` records, then run:

```bash
python3 scripts/estimate_m3dgr_handeye.py \
  --manifest /path/to/handeye_runs.yaml \
  --output-json /path/to/m3dgr_handeye.json
```

Per-run estimates must differ by less than 2 cm and 1 degree. A passing report
is still `handeye_consistent_unconfirmed`. Pass it to the aligner with
`--gt-body-to-imu-report`; set `--rigid-body-frame-status confirmed` only after
the physical frame is confirmed externally.

## 5. Build Right-SE(3) Labels

The label is aligned to the sequence last frame:

```text
DeltaT = inverse(T_est) * T_ref
xi_gt = Log_SE3(DeltaT)
y = [delta_theta_x, delta_theta_y, delta_theta_z,
     delta_rho_x, delta_rho_y, delta_rho_z]
```

`delta_rho` is the translational component of the SE(3) logarithm in the IMU
local frame. It is not world-frame `p_ref-p_est`, and rotations are not Euler angles.

Pipeline-only Pilot audit:

```bash
python3 scripts/build_m3dgr_mocap_dataset.py \
  --manifest config/m3dgr_dataset_manifest.yaml \
  --output-dir Log/datasets/m3dgr_pilot \
  --mode pilot \
  --pilot-sequence Varying-illu02 \
  --allow-alignment-candidate
```

Pilot mode deliberately does not compute normalization from the test bag.

After every sequence has `trusted_gt` and the manifest paths are filled:

```bash
python3 scripts/build_m3dgr_mocap_dataset.py \
  --manifest config/m3dgr_dataset_manifest.yaml \
  --output-dir Log/datasets/m3dgr_mocap \
  --mode formal
```

Formal mode enforces the frozen split, one shared frame convention/extrinsic,
time-offset stability, bag-level isolation, train-only normalization, and
`Exp(Log(.))` reconstruction. It never filters a valid label using runtime
rotation/translation safety thresholds.

## 6. Current Verified State

Verified on 2026-08-08:

- all six synthetic regression tests pass;
- ROS2 launch arguments load and `fastlivo_mapping` starts in LIO export mode;
- the package builds successfully with ONNX Runtime enabled;
- the public `Varying-illu02.txt` has 43,550 poses over 146.49 s;
- its SHA256 is `293f0ef2...535d0f`;
- timestamps are strictly increasing;
- maximum raw quaternion norm error is `7.858e-7`, below the strict `1e-6` gate.

Not yet completed:

- the 1.75 GB Pilot rosbag is not present locally;
- no real FAST-LIVO2 Pilot CSV has been exported;
- the Mocap rigid-body physical definition is not confirmed;
- therefore no artifact is currently eligible as trusted M3DGR training GT;
- causal Mamba/baseline training remains gated and has not started.
