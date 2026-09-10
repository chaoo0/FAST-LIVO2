# M3DGR Phase 1: ROS 2 FAST-LIVO2 baseline report

Date: 2026-09-10

Status: **not accepted; stopped before LIVO and before Phase 2**

## 1. Repository state and isolation

| Check | Result |
|---|---|
| Development remote | `origin = https://github.com/chaoo0/FAST-LIVO2.git` |
| Phase branch | `phase1/m3dgr-baseline` |
| Branch point | `d6e22ad4e1ee478e41050cfa83a3c35e98d00ee5` |
| Existing modified commit | `10a0185ca8bbb3ae09d443cf3e76eead1b1fb271`, preserved in Git history |
| Phase 1 core diff | no changes under `src/` or `include/` relative to `d6e22ad` |

- 【事实】Before creating the Phase 1 branch, `10a0185` was checked out on `ros2-restart` with a clean worktree. It changes six estimator files with 234 insertions and 21 deletions.
- 【事实】`10a0185` does not currently link: `LIVMapper::writeLioDiagnostics(...)` is declared/called but has no linked definition. Its modifications are incomplete and were not imported into this phase.
- 【事实】The phase branch was created directly at `d6e22ad`; existing Git history was not rewritten or discarded.

The Phase 1 additions are limited to M3DGR YAML/launch files, audit/evaluation/run scripts, tests, documentation, and CMake installation entries.

## 2. Build and static checks

- 【事实】ROS 2 Humble build command:

  ```bash
  source /opt/ros/humble/setup.bash
  colcon build --symlink-install --packages-select fast_livo \
    --cmake-clean-cache --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  ```

- 【事实】The package built successfully. The emitted PCL/CMake developer-policy warnings were non-fatal.
- 【事实】The M3DGR launch description resolved successfully with `ros2 launch ... --show-args`.
- 【事实】Two synthetic position-evaluation unit tests passed: known rigid-frame recovery without scale fitting, and duplicate-GT timestamp averaging.

Build success only establishes compilability. It does not establish numerical correctness or Phase 1 acceptance.

## 3. Fixed run and evaluation protocol

Three valid trials used:

- `Outdoor01_ros2`, normal speed (`--rate 1.0`);
- only `/livox/mid360/lidar` and `/livox/mid360/imu` replayed;
- images, RViz, and PCD saving disabled;
- the same commit, parameter files, host, and launch script;
- separate ROS domain IDs and output directories;
- estimator TUM output copied only after playback and a bounded stable-count drain.

One earlier run overlapped a CPU-heavy full-bag Python deserialization audit. It was invalidated before metric comparison, retained under `invalid_run1_concurrent_audit`, and is not one of the three trials.

Position evaluation protocol:

- quaternion order in TUM files: `qx qy qz qw`;
- estimate convention: estimator world-frame pose of the IMU state;
- GT duplicate timestamps: stable sort then mean position per timestamp;
- time association: linear GT position interpolation at estimate timestamps;
- maximum accepted GT interpolation bracket: 0.25 s;
- alignment: one fixed-scale rigid SE(3) fit on matched positions; no Sim(3) scale fitting;
- ATE: aligned position RMSE;
- RPE: world-frame displacement error at 1, 5, and 10 s;
- horizon match tolerance: 0.06 s.

The machine-readable protocol is fixed in `config/m3dgr_position_eval.json` and is consumed with `evaluate_m3dgr_position.py --config ...`.

【事实】The GT rotations are all identity, so results below are position metrics, not full SE(3) metrics. The evaluator fails to claim rotational RPE explicitly.

## 4. Three-run results

| Metric | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Input LiDAR messages | 4,116 | 4,116 | 4,116 |
| Input IMU messages | 82,312 | 82,312 | 82,312 |
| Output poses / LIO updates | 4,113 | 4,113 | 4,113 |
| First output timestamp | 1735888008.419853 | same | same |
| Last output timestamp | 1735888419.619859 | same | same |
| Matched GT poses | 4,111 | 4,111 | 4,111 |
| Position ATE RMSE (m) | 0.274203 | 0.280398 | 0.280398 |
| Position RPE 1 s RMSE (m) | 0.031750 | 0.032281 | 0.032281 |
| Position RPE 5 s RMSE (m) | 0.072853 | 0.073063 | 0.073063 |
| Position RPE 10 s RMSE (m) | 0.112948 | 0.113641 | 0.113641 |
| Mean frame time (ms) | 19.256 | 19.336 | 19.504 |
| p99 frame time (ms) | 35.339 | 35.612 | 37.708 |
| Maximum frame time (ms) | 145.706 | 109.156 | 115.720 |
| Playback-to-output wall duration (s) | 418 | 418 | 418 |

- 【事实】All three trials produced exactly the same output count and timestamps, with 4,113 updates from 4,116 LiDAR messages. Logs report three initial `No point!!!` updates and no LiDAR/IMU timestamp loopback or out-of-sync warning.
- 【事实】Runs 2 and 3 have byte-identical trajectory files. Their pairwise position RMSE is 0 and rotation RMSE is about `2.05e-8` rad from numerical quaternion comparison.
- 【事实】Run 1 versus either Run 2 or Run 3 has pairwise position RMSE 0.258149 m, maximum 0.469349 m, rotation RMSE 0.003566 rad, and maximum 0.007654 rad.
- 【事实】Position ATE sample coefficient of variation is 1.285%; population CV is 1.049%. Both exceed the predeclared 1% acceptance threshold. Position-RPE sample CV is 0.955%, 0.166%, and 0.353% at 1, 5, and 10 s respectively.
- 【事实】All three trials reached within one scan period of the final LiDAR timestamp. Mean and p99 processing time were below the approximately 100 ms scan period, while one or more individual frames exceeded it.
- 【推断】Final timestamp coverage, bounded drain, and 418 s wall duration for a 411.6 s bag are evidence against sustained queue growth. Exact queue length was not instrumented, so “no queue growth” is not proven.

Artifacts are stored outside the source repository at `/home/liu/fast_livo2/results/phase1_m3dgr/Outdoor01`. SHA-256:

| Artifact | SHA-256 |
|---|---|
| Run 1 trajectory | `255c53f037b9c242a69d9a12b4ef82f0d0fbd49a8878d9c2bd8539c1b606e421` |
| Run 2 trajectory | `bc0f5c9483a82e43fe11e39433fe973e1a887781cc7f5dd5adc1729444b5e227` |
| Run 3 trajectory | `bc0f5c9483a82e43fe11e39433fe973e1a887781cc7f5dd5adc1729444b5e227` |
| Three-run JSON summary | `d83e753e806b16be5447379df981cc0888f34162b1d659aa8ae463b622884e33` |

## 5. Repeatability failure analysis

- 【事实】Run 1 IMU initialization progress was logged as 3.3%, 30.0%, and 96.7%, with initial gravity approximately `[4.4578, 0.2947, -8.7337]`. Runs 2/3 logged 3.3%, 33.3%, and 100%, with gravity approximately `[4.4587, 0.2943, -8.7332]`.
- 【事实】The baseline IMU callback discards IMU messages until the first LiDAR timestamp has been observed. IMU initialization then accumulates whichever grouped IMU samples reach each initialization update.
- 【事实】Run 1 executed 20,501 raw feature-iteration log entries; Runs 2/3 executed 20,510, so divergence existed from initialization onward rather than arising only in the evaluator.
- 【推断】Cross-topic delivery/grouping around the first LiDAR frame changed the number of IMU samples used for initialization. The resulting gravity difference propagated into a different trajectory. This is the strongest current explanation, but proving the scheduler-level cause requires instrumentation that would modify core runtime behavior.
- 【未知】Whether deterministic buffering alone is sufficient to bring ATE CV below 1%, and whether the same behavior occurs on other hosts/RMW implementations, has not been experimentally established.

This is a candidate ROS 2 migration/runtime-contract defect for Phase 2. It is deliberately not fixed in the Phase 1 configuration branch.

## 6. Additional unresolved issues

1. 【事实】On bounded SIGINT cleanup after all data and trajectory output had completed, the mapping process exited with signal 11 in all three trials. The stored trajectories were already flushed, but clean shutdown is not achieved.
2. 【事实】Available outdoor GT supports position-only evaluation; full rotational/SE(3) ATE and RPE are unavailable.
3. 【推断】RTK antenna versus estimator IMU reference-point mismatch can contribute to position error because a time-varying lever-arm correction cannot be reconstructed without GT orientation.
4. 【未知】Residual sensor time offset is not independently calibrated. The official dataset states software synchronization but does not provide a sequence-specific bound.
5. 【事实】Exact runtime queue depth and dropped DDS samples were not instrumented. Counts and end-time coverage are indirect checks only.
6. 【事实】The visual calibration and LIVO execution were not validated because the LIO stability gate failed first.

## 7. Acceptance decision

| Phase 1 gate | Status | Evidence |
|---|---|---|
| ROS 2 package builds | PASS | clean `d6e22ad` derivative builds |
| M3DGR MID-360 bag readable | PASS | metadata, full serialized traversal, Corridor02 full deserialization |
| Configuration has traceable sources | PASS | official calibration and M3DGR adaptation mapped in data report |
| LIO runs end to end | PASS | three complete Outdoor01 trajectories |
| Three output counts/timestamps agree | PASS | 4,113, identical timestamps |
| Numerical repeatability <= 1% | **FAIL** | ATE CV 1.285% sample / 1.049% population |
| GT evaluation flow verified | PARTIAL | position protocol tested; no trustworthy 6DoF GT |
| No algorithm behavior changes | PASS | no `src/` or `include/` diff from `d6e22ad` |
| LIVO validated | NOT RUN | prohibited by failed LIO stability gate |

【结论】Phase 1 is implemented far enough to build, read data, run LIO, and produce a reproducible position-only baseline, but it has **not passed acceptance**. The repeatability threshold is missed and the current GT cannot support full SE(3) validation. Per the staged project rule, no LIVO experiment, official-vs-ROS2 bug fix, or Mamba work is started from this result.

The next authorized work, after user review, should remain within Phase 1/its blocking interface: decide whether to (a) instrument and diagnose initialization nondeterminism as a narrowly scoped prerequisite, or (b) revise the Phase 1 stability/GT acceptance contract with explicit scientific justification. The threshold must not be relaxed post hoc merely to pass.
