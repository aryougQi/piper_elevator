#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

# Keep the standalone visualizer in the same ROS domain as the documented
# simulation workflow, while allowing callers to override it explicitly.
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}" docker compose run --rm piper_ros2 bash -lc '
    set -e
    source /workspace/ros2_ws/install/setup.bash
    exec ros2 launch piper_elevator_app piper_pika_moveit.launch.py \
        external_hardware:=false \
        use_sim_time:=false \
        use_rviz:=true \
        start_moveit_servo:=false \
        "$@"
' bash "$@"
