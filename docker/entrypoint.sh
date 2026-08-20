#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash

# Qt tools such as rqt_image_view run inside the container through the host X
# socket. Use Mesa software rendering so they do not try to load a mismatched
# host nouveau driver, and provide the private runtime directory Qt expects.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-root}"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"

if [[ -f /workspace/ros2_ws/install/setup.bash ]]; then
    source /workspace/ros2_ws/install/setup.bash
fi

exec "$@"
