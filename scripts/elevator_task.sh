#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

simulation_enabled=true
gui_enabled=true
gazebo_gui_argument=false
simulation_mode_argument=false
sam2_argument=false
sam2_debug_argument=false
for argument in "$@"; do
    if [[ "${argument}" == simulation_mode:=* ]]; then
        simulation_mode_argument=true
    fi
    if [[ "${argument}" == enable_sam2:=* ]]; then
        sam2_argument=true
    fi
    if [[ "${argument}" == sam2_debug_image:=* ]]; then
        sam2_debug_argument=true
    fi
    if [[ "${argument}" == "simulation_mode:=false" ]]; then
        simulation_enabled=false
    elif [[ "${argument}" == "gazebo_gui:=false" ]]; then
        gui_enabled=false
        gazebo_gui_argument=true
    elif [[ "${argument}" == gazebo_gui:=* ]]; then
        gazebo_gui_argument=true
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

# elevator_task.launch.py defaults to headless mode so CI and containers do
# not crash without X11. This convenience script is the interactive entry
# point, so preserve its historical behavior and explicitly request the GUI
# unless the caller supplied gazebo_gui:=... themselves.
launch_arguments=("$@")
if ! ${simulation_mode_argument}; then
    launch_arguments+=(simulation_mode:=true)
fi
if ! ${sam2_argument}; then
    launch_arguments+=(enable_sam2:=true)
fi
if ! ${sam2_debug_argument}; then
    launch_arguments+=(sam2_debug_image:=true)
fi
if ! ${gazebo_gui_argument}; then
    launch_arguments+=(gazebo_gui:=true)
fi

ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}" docker compose run --rm piper_ros2 bash -lc '
    source /workspace/ros2_ws/install/setup.bash
    exec ros2 launch piper_elevator_app elevator_task.launch.py "$@"
' bash "${launch_arguments[@]}"
