#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" docker compose run --rm piper_ros2 bash -lc '
    set -eo pipefail
    source /workspace/ros2_ws/install/setup.bash
    set -u

    launch_log=/tmp/piper_gazebo_hardware.log
    ros2 launch piper_elevator_gazebo gazebo_hardware.launch.py \
        gui:=false verbose:=2 >"${launch_log}" 2>&1 &
    launch_pid=$!

    cleanup() {
        if kill -0 "${launch_pid}" 2>/dev/null; then
            kill -TERM "${launch_pid}" 2>/dev/null || true
            for _ in $(seq 1 20); do
                if ! kill -0 "${launch_pid}" 2>/dev/null; then
                    break
                fi
                sleep 0.25
            done
            kill -KILL "${launch_pid}" 2>/dev/null || true
        fi
        wait "${launch_pid}" 2>/dev/null || true
    }
    trap cleanup EXIT

    gazebo_share="$(ros2 pkg prefix piper_elevator_gazebo)/share/piper_elevator_gazebo"
    expanded="$(xacro "${gazebo_share}/urdf/piper_pika_gazebo.urdf.xacro")"
    grep -Fq "gz_ros2_control/GazeboSimSystem" <<<"${expanded}"
    if grep -Fq "mock_components/GenericSystem" <<<"${expanded}"; then
        echo "Gazebo description unexpectedly contains GenericSystem." >&2
        exit 1
    fi

    python3 \
        /workspace/ros2_ws/src/piper_elevator_gazebo/test/virtual_hardware_probe.py
    if ! grep -Fq "position_proportional_gain has been set to: 0.1" \
        "${launch_log}"; then
        cat "${launch_log}" >&2
        echo "Gazebo position-interface runtime gain changed unexpectedly." >&2
        exit 1
    fi
    if grep -Fq "Unable to find file with URI" "${launch_log}"; then
        cat "${launch_log}" >&2
        echo "Gazebo could not resolve one or more official model resources." >&2
        exit 1
    fi
    echo "OK: hardware-only Gazebo launch has three active controllers and no core nodes."
'
