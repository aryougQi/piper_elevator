#!/usr/bin/env bash
set -euo pipefail

# Explicit test entry for the existing nominal hand-eye values. This wrapper
# deliberately does not write URDF, driver zero offsets, or joint offsets.
# Hardware execution remains disabled unless the caller explicitly overrides it.
exec "$(dirname -- "${BASH_SOURCE[0]}")/button_approach_real.sh" \
  publish_camera_tf:=true \
  camera_calibration_valid:=true \
  hardware_commands_enabled:=false \
  allow_execution:=false \
  "$@"
