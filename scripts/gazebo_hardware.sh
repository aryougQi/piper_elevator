#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

gui_enabled=true
for argument in "$@"; do
    if [[ "${argument}" == "gui:=false" ]]; then
        gui_enabled=false
    fi
done

if ${gui_enabled}; then
    if [[ -z "${DISPLAY:-}" ]]; then
        echo "Gazebo GUI requires DISPLAY; use gui:=false for headless mode." >&2
        exit 1
    fi
    if [[ -z "${XAUTHORITY:-}" || ! -f "${XAUTHORITY}" ]]; then
        echo "Gazebo GUI cannot read the current Xauthority file." >&2
        echo "Log into the desktop session again, or use gui:=false." >&2
        exit 1
    fi
fi

docker_args=(run --rm -e IGN_PARTITION)
if [[ -n "${PANEL_SIM_CONTAINER_NAME:-}" ]]; then
    docker_args+=(--name "${PANEL_SIM_CONTAINER_NAME}")
fi
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" docker compose "${docker_args[@]}" \
  -e IGN_GAZEBO_RESOURCE_PATH=/workspace/ros2_ws/src/piper_elevator_gazebo/sim_variants/models:/workspace/ros2_ws/src/piper_elevator_gazebo/models \
  -e GZ_SIM_RESOURCE_PATH=/workspace/ros2_ws/src/piper_elevator_gazebo/sim_variants/models:/workspace/ros2_ws/src/piper_elevator_gazebo/models \
  piper_ros2 bash -lc '
    source /workspace/ros2_ws/install/setup.bash
    exec ros2 launch piper_elevator_gazebo gazebo_hardware.launch.py "$@"
' bash "$@"
