# Diagnostics

This is the dedicated location for diagnostic scripts and generated evidence.
In Docker it is mounted at `/workspace/ros2_ws/diagnostics`.

## Layout

- `scripts/`: standalone IK, motion-quality, post-execution, and Servo diagnostics.
- `data/`: JSON reports, NPZ replay inputs, screenshots, and capture directories.
  Generated data is retained locally and ignored by Git.
- Package tools remain in `../src/piper_elevator_app/scripts/`; their default
  output also goes to `data/`.

Default paths are based on script locations, not the current working directory.
Explicit input/output arguments still take precedence. For a new experiment, use
a descriptive filename or run subdirectory under `data/` so baseline evidence is
not overwritten. Some tools reject an existing output; choose a new name.

## Retained Evidence

| Files in `data/` | Purpose |
| --- | --- |
| `coarse_*` | Joint-limit, planning, repeatability, and camera-view comparisons |
| `ik_candidate_*` | Frozen observations and production IK validation |
| `approach_quality_benchmark.json`, `approach_quality_matrix_verified.json` | Motion-quality baseline and corrected multi-scenario results |
| `post_execution_*`, `rgbd_diagnostic.json` | RGB-D and post-motion geometry evidence |
| `servo_handoff_diagnostic.json` | Servo handoff and command observations |
| `surface_support_diagnostic.*`, `surface_support_diagnostic_color.png` | Original depth replay dataset and reference image |
| `vision_stability_*`, `vision_surface_parameter_sweep.json` | Vision datasets and algorithm/parameter comparisons |
| `vision_detector_*`, `vision_reload_idle_check.json` | Detector reload record, runtime parameter snapshot, and live log |
| `updown_diagnostic/` | Direction-label captures and before/after evidence used by regression tests |
| `coarse_detection_loss/` | Healthy raw detections versus frozen world-point projection after coarse motion |
| `handeye_frame_comparison_20260908_140832.json`, `handeye_frame_audit/` | Calibration/runtime URDF, FK, extrinsics and pose coverage comparison |
| `handeye_sample_integrity_20260908.json`, `handeye_resampling_audit.json` | Previous-pose TF records and offline fit sensitivity; no deployed calibration changes |

Reports describe the recorded scene and software state; they do not establish
current robot readiness. Historical JSON metadata has been preserved unchanged.
Old `/workspace/ros2_ws/<file>` paths in reports now refer to
`/workspace/ros2_ws/diagnostics/data/<file>`. Basename-only references are relative
to `data/`; image references inside `updown_diagnostic/` remain relative to it.
The live detector log was renamed within the same filesystem, so its existing
open file handle continues writing to `data/vision_detector_reload.log`. The
runtime parameter snapshot is now `data/vision_detector_runtime_backup.yaml`;
use that path when explicitly reusing the snapshot in a future launch.

## Examples

Run from `ros2_ws` with the ROS workspace environment loaded:

```bash
python3 src/piper_elevator_app/scripts/diagnose_rgbd.py --seconds 15 \
  --output diagnostics/data/new_run/rgbd.json
python3 src/piper_elevator_app/scripts/diagnose_surface_support.py \
  --replay diagnostics/data/surface_support_diagnostic.npz \
  --output-prefix diagnostics/data/new_run/surface_replay
python3 src/piper_elevator_app/scripts/benchmark_surface_geometry.py \
  diagnostics/data/vision_stability_full_context.npz \
  --output diagnostics/data/new_run/geometry_benchmark.json
```

Offline replay does not require a running ROS graph, but scripts may still import
ROS workspace libraries. Live diagnostic requirements are documented in the
individual scripts and `../../DEBUG_COMMANDS.md`.

## Cleanup on 2026-09-08

- Removed `ros2_ws/core`: an old `move_group` crash dump, about 256 MiB on disk.
- Removed `approach_quality_matrix.json`: the initial report contained diagnostic
  script errors and was superseded by `approach_quality_matrix_verified.json`.
- Removed `vision_stability_runtime_health.json`: an isolated counter snapshot
  without capture time, duration, or replay context.
- Preserved the original contents of all other diagnostic datasets and images.
