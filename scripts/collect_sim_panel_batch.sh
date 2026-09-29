#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_root="$(cd -- "${project_root}/../Yolo_Train/sim-dataset" && pwd)"
mkdir -p -- "${dataset_root}/Raw/batch_logs"
lights=(${LIGHTS:-L0 L1 L2 L3 L4 L5})
domain="${ROS_DOMAIN_ID:-78}"
sim_name=""
sim_pid=""

cleanup() {
    if [[ -n "${sim_name}" ]]; then
        docker stop -t 5 "${sim_name}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${sim_pid}" ]]; then
        wait "${sim_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

for light in "${lights[@]}"; do
    [[ "${light}" =~ ^L[0-5]$ ]] || { echo "Invalid light: ${light}" >&2; exit 2; }
    partition="panel_capture_${light}_${domain}"
    sim_name="piper-panel-capture-${light,,}-$$"
    log_prefix="${dataset_root}/Raw/batch_logs/$(date +%Y%m%d_%H%M%S)_${light}"
    echo "Starting ${light} in ROS domain ${domain} (log: ${log_prefix})"
    ROS_DOMAIN_ID="${domain}" IGN_PARTITION="${partition}" \
      PANEL_SIM_CONTAINER_NAME="${sim_name}" \
      "${project_root}/scripts/gazebo_hardware.sh" gui:=false \
      "world:=/workspace/ros2_ws/src/piper_elevator_gazebo/sim_variants/worlds/light_${light}.sdf" \
      >"${log_prefix}_gazebo.log" 2>&1 &
    sim_pid=$!
    sleep 8
    if ! kill -0 "${sim_pid}" 2>/dev/null; then
        echo "Gazebo exited early; see ${log_prefix}_gazebo.log" >&2
        exit 1
    fi
    ROS_DOMAIN_ID="${domain}" IGN_PARTITION="${partition}" \
      "${project_root}/scripts/collect_sim_panel_dataset.sh" --execute \
      --variants v0,v1,v2,v3,v4 --jitters 2 --seed 20260923 \
      --joint1-angles=-12,-6,0,6,12 --joint5-angles=-5,0,5 \
      --move-seconds 1.2 --settle-seconds 0.2 --no-depth-check \
      --scene-id "${light}" >"${log_prefix}_collector.log" 2>&1 || {
        cat "${log_prefix}_collector.log" >&2
        exit 1
      }
    tail -n 2 "${log_prefix}_collector.log"
    cleanup
    sim_name=""
    sim_pid=""
done
