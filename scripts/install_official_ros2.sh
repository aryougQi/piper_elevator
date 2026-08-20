#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${project_root}/ros2_ws/src"

mkdir -p "${source_dir}"

install_repository() {
    local name="$1"
    local url="$2"
    local repository_dir="${source_dir}/${name}"

    if [[ -d "${repository_dir}/.git" ]]; then
        echo "${name} already exists; keeping the current checkout."
        git -C "${repository_dir}" submodule update --init --recursive
        return
    fi

    if [[ -e "${repository_dir}" ]]; then
        echo "Error: ${repository_dir} exists but is not a Git checkout." >&2
        exit 1
    fi

    git clone \
        --depth 1 \
        --branch ros2 \
        --recurse-submodules \
        --shallow-submodules \
        "${url}" \
        "${repository_dir}"
}

install_repository agx_arm_ros https://github.com/agilexrobotics/agx_arm_ros.git
install_repository pika_ros https://github.com/agilexrobotics/pika_ros.git
