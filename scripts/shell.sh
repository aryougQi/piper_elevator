#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

docker compose run --rm piper_ros2 bash -lc '
    runtime_dir="${XDG_RUNTIME_DIR:-/tmp/runtime-root}"
    mkdir -p "${runtime_dir}"
    chmod 700 "${runtime_dir}"
    exec bash
'
