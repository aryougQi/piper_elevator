#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

mode=both
runs=5
button=3
timeout=240
log_dir="${project_root}/ros2_ws/diagnostics/data/stability"
ros_domain_id="${ROS_DOMAIN_ID:-42}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) mode="$2"; shift 2 ;;
        --runs) runs="$2"; shift 2 ;;
        --button) button="$2"; shift 2 ;;
        --timeout) timeout="$2"; shift 2 ;;
        --log-dir) log_dir="$2"; shift 2 ;;
        -h|--help)
            cat <<'EOF'
Usage: ./scripts/stability_test.sh [options]

  --mode stable|yolo|both   Test deterministic target, current YOLO, or both
  --runs N                  Trials per mode (default: 5)
  --button NAME             Button class (default: 3)
  --timeout SECONDS         Per-trial timeout (default: 240)
  --log-dir DIR             CSV and launch logs (default: test_logs/stability)
EOF
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "${mode}" in stable|yolo|both) ;; *) echo "--mode must be stable, yolo, or both" >&2; exit 2 ;; esac
mkdir -p "${log_dir}"

run_mode() {
    local test_mode="$1"
    local stable_target=false
    [[ "${test_mode}" == stable ]] && stable_target=true
    local launch_log="${log_dir}/launch_${test_mode}_$(date +%Y%m%d_%H%M%S).log"
    local stack_pid=''

    cleanup() {
        if [[ -n "${stack_pid}" ]] && kill -0 "${stack_pid}" 2>/dev/null; then
            kill -TERM "${stack_pid}" 2>/dev/null || true
            wait "${stack_pid}" 2>/dev/null || true
        fi
    }
    trap cleanup RETURN

    echo "Starting ${test_mode} stack; launch log: ${launch_log}"
    ROS_DOMAIN_ID="${ros_domain_id}" docker compose run --rm -T \
        -e ROS_DOMAIN_ID="${ros_domain_id}" \
        -v "${project_root}/scripts:/workspace/project_scripts:ro" \
        piper_ros2 bash -lc "
            source /workspace/ros2_ws/install/setup.bash
            exec ros2 launch piper_elevator_app elevator_task.launch.py \\
              simulation_mode:=true gazebo_gui:=false use_rviz:=false \\
              stable_target_mode:=${stable_target}
        " >"${launch_log}" 2>&1 &
    stack_pid=$!

    set +e
    ROS_DOMAIN_ID="${ros_domain_id}" docker compose run --rm -T \
        -e ROS_DOMAIN_ID="${ros_domain_id}" \
        -v "${project_root}/scripts:/workspace/project_scripts:ro" \
        -v "${log_dir}:/workspace/test_logs" \
        piper_ros2 bash -lc "
            source /workspace/ros2_ws/install/setup.bash
            exec python3 /workspace/project_scripts/stability_test.py \\
              --mode ${test_mode} --runs ${runs} \\
              --button ${button} --timeout ${timeout} \\
              --log-dir /workspace/test_logs
        "
    local result=$?
    set -e
    if [[ ${result} -ne 0 ]]; then
        echo "${test_mode} stability test failed; inspect ${launch_log}" >&2
    fi
    return ${result}
}

overall=0
case "${mode}" in
    stable) run_mode stable || overall=1 ;;
    yolo) run_mode yolo || overall=1 ;;
    both)
        run_mode stable || overall=1
        run_mode yolo || overall=1
        ;;
esac
exit ${overall}
