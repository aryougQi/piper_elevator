#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

docker compose run --rm piper_ros2 bash -lc '
    source /workspace/ros2_ws/install/setup.bash
    ros2 launch agx_arm_moveit demo.launch.py \
        arm_type:=piper \
        effector_type:=none
'

