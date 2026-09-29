#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_root="$(cd -- "${project_root}/../Yolo_Train/sim-dataset" && pwd)"
mkdir -p -- "${dataset_root}/Raw"

cd -- "${project_root}"
exec docker compose run --rm \
  -v "${dataset_root}:/workspace/sim-dataset" \
  -e IGN_PARTITION \
  piper_ros2 bash -lc '
    source /workspace/ros2_ws/install/setup.bash
    exec python3 /workspace/ros2_ws/src/piper_elevator_app/scripts/collect_sim_panel_dataset.py "$@"
  ' bash "$@"
