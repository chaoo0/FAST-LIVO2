# MambaPose ONNX Interface Reference

This document defines the stable C++/ONNX contract. For runtime diagnosis use
`mamba_pose_debug_checklist.md`; for historical model results use
`mamba_pose_baseline_history.md`.

## Contract

The FAST-LIVO2 backend supplies a single raw basic18 sequence:

```text
input name: input
input shape: [10, 18]
output name: output
output shape: [6]
```

The basic18 order is:

```text
pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,rot_w,
vel_x,vel_y,vel_z,bias_g_x,bias_g_y,bias_g_z,
bias_a_x,bias_a_y,bias_a_z,effective_feature_num,avg_residual
```

Feature normalization is an offline-training concern. A model used by C++ must
either accept raw basic18 directly or embed the saved normalization in a
wrapper. The repository convention is the latter, named `*_raw_input.onnx`.
Normalized-input models are offline-only.

## Model and YAML rules

- Test-only models/YAMLs: `mamba_pose_onnx_normal_test.yaml`,
  `mamba_pose_onnx_safety_test.yaml`, and `mamba_pose_onnx_reject_test.yaml`.
  They exist to verify the backend and safety limits, never to export data.
- Formal CSV export uses dummy backend and a run-specific output path.
- All learned baseline and future GT models start with
  `mamba_pose/apply_correction_en=false`.
- A right-SE(3) GT model must remain observe-only until the C++ state-apply
  contract is migrated and validated.

## Minimal smoke test

1. Verify the model graph exposes exactly the names/shapes above.
2. Point `mamba_pose/model_path` to an absolute `*_raw_input.onnx` path.
3. Start MID360 with the corresponding observe-only YAML.
4. After ten valid frames, require `model_loaded=true`, `session_ready=true`,
   `io_name_ready=true`, `inference_success=true`, and `applied=false`.

For a controlled nonzero/safety test, use the exact command in
`mamba_pose_safety_test.md`.
