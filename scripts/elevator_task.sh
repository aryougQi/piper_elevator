#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

simulation_enabled=true
gui_enabled=true
for argument in "$@"; do
    if [[ "${argument}" == "simulation_mode:=false" ]]; then
        simulation_enabled=false
    elif [[ "${argument}" == "gazebo_gui:=false" ]]; then
        gui_enabled=false
    fi
done

if ${simulation_enabled} && ${gui_enabled}; then
    if [[ -z "${DISPLAY:-}" ]]; then
        echo "Gazebo GUI requires DISPLAY; use gazebo_gui:=false." >&2
        exit 1
    fi
    if [[ -z "${XAUTHORITY:-}" || ! -f "${XAUTHORITY}" ]]; then
        echo "Gazebo GUI cannot read the current Xauthority file." >&2
        echo "Log in again, or use gazebo_gui:=false." >&2
        exit 1
    fi
fi

ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}" docker compose run --rm piper_ros2 bash -lc '
    source /workspace/ros2_ws/install/setup.bash
    exec ros2 launch piper_elevator_app elevator_task.launch.py "$@"
' bash "$@"
