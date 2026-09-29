#!/usr/bin/env bash
# Press every panel button several times through the elevator task state
# machine and report per-button stability.
#
#   ./scripts/test_button_stability.sh --execute
#   ./scripts/test_button_stability.sh --execute --buttons 1,2,3 --rounds 5
#   ./scripts/test_button_stability.sh --execute --attach   # stack already up
#
# By default this script brings up the complete headless simulation
# (Gazebo + MoveIt + detector + planner + visual servo + press + task state
# machine) for the duration of the run and tears it down afterwards.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

run_in_container() {
    docker compose run --rm -T \
        -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}" \
        -e BUTTON_STABILITY_ATTACH="${attach}" \
        -v "${project_root}/scripts:/workspace/project_scripts:ro" \
        -v "${project_root}/test_logs:/workspace/test_logs" \
        piper_ros2 bash -lc '
            set -eo pipefail
            source /opt/ros/humble/setup.bash
            source /workspace/ros2_ws/install/setup.bash

            sim_pid=""
            sim_log=/workspace/test_logs/button_stability_sim.log
            cleanup() {
                if [[ -n "${sim_pid}" ]] && kill -0 "${sim_pid}" 2>/dev/null; then
                    kill -TERM -"${sim_pid}" 2>/dev/null \
                        || kill -TERM "${sim_pid}" 2>/dev/null || true
                    for _ in $(seq 1 40); do
                        kill -0 "${sim_pid}" 2>/dev/null || break
                        sleep 0.25
                    done
                    kill -KILL -"${sim_pid}" 2>/dev/null || true
                fi
            }
            trap cleanup EXIT

            if [[ "${BUTTON_STABILITY_ATTACH}" != "1" ]]; then
                echo "Starting headless elevator_task stack (log: ${sim_log})"
                setsid ros2 launch piper_elevator_app elevator_task.launch.py \
                    gazebo_gui:=false use_rviz:=false \
                    >"${sim_log}" 2>&1 &
                sim_pid=$!
            else
                echo "Attaching to the already running task stack."
            fi

            python3 /workspace/project_scripts/button_stability_test.py "$@"
        ' bash "$@"
}

# Drop a leading --attach so it is consumed here instead of argparse.
attach=0
forwarded=()
for argument in "$@"; do
    if [[ "${argument}" == "--attach" ]]; then
        attach=1
    else
        forwarded+=("${argument}")
    fi
done

if [[ "${ROS_DISTRO:-}" != "humble" ]] \
    || [[ ! -f /opt/ros/humble/setup.bash ]]; then
    cd "${project_root}"
    mkdir -p "${project_root}/test_logs"
    if (( ${#forwarded[@]} )); then
        run_in_container "${forwarded[@]}"
    else
        run_in_container
    fi
    exit $?
fi

source "${project_root}/ros2_ws/install/setup.bash"
exec python3 \
    "${project_root}/scripts/button_stability_test.py" "${forwarded[@]}"
