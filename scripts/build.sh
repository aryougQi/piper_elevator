#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

if [[ ! -d ros2_ws/src/agx_arm_ros/.git ]]; then
    echo "Official agx_arm_ros is missing." >&2
    echo "Run ./scripts/install_official_ros2.sh first." >&2
    exit 1
fi

if [[ ! -d ros2_ws/src/pika_ros/.git ]]; then
    echo "Official pika_ros is missing." >&2
    echo "Run ./scripts/install_official_ros2.sh first." >&2
    exit 1
fi

docker compose run --rm piper_ros2 bash -lc '
    set -e
    cd /workspace/ros2_ws
    # The two MoveIt components are optional and are not published in the
    # configured Humble apt repository. Pika declares "serial", but its
    # current ROS 2 CMake files do not actually use that library.
    rosdep install \
        --from-paths src \
        --ignore-src \
        --rosdistro humble \
        --skip-keys "ament_python launch_pytest moveit_ros_perception warehouse_ros_mongo serial" \
        -r -y
    colcon build \
        --symlink-install \
        --cmake-args -DBUILD_TESTING=OFF
'
