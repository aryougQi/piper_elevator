#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

ROS_DOMAIN_ID=0 docker compose run --rm piper_ros2 bash -lc '
    source /workspace/ros2_ws/install/setup.bash
    exec ros2 launch piper_elevator_app button_approach_real.launch.py "$@"
' bash "$@"
