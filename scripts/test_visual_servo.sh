#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${ROS_DISTRO:-}" != "humble" ]] \
    || [[ ! -f /opt/ros/humble/setup.bash ]]; then
    cd "${project_root}"
    mkdir -p "${project_root}/test_logs"
    exec docker compose run --rm -T \
        -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}" \
        -v "${project_root}/scripts:/workspace/project_scripts:ro" \
        -v "${project_root}/test_logs:/workspace/test_logs" \
        piper_ros2 bash -lc '
            source /opt/ros/humble/setup.bash
            source /workspace/ros2_ws/install/setup.bash
            exec python3 \
                /workspace/project_scripts/visual_servo_repeatability_test.py \
                "$@"
        ' bash "$@"
fi

source "${project_root}/ros2_ws/install/setup.bash"
exec python3 \
    "${project_root}/scripts/visual_servo_repeatability_test.py" "$@"
