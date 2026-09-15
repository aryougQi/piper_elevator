# Diagnostic artifacts

- Store diagnostic scripts, captures, screenshots, replay inputs, and reports under
  `ros2_ws/diagnostics/`. Do not create these artifacts in the project root or
  directly in `ros2_ws/`.
- Put standalone diagnostic scripts in `ros2_ws/diagnostics/scripts/` and generated
  data in `ros2_ws/diagnostics/data/`. Existing package diagnostic tools may stay
  in their package, but their default output must use this data directory.
- Resolve default paths from the script location so host and Docker commands work
  regardless of the current directory. Preserve explicit output arguments.
- Use a descriptive run name or subdirectory for new experiments, and preserve
  useful baseline captures and reports. Keep paired JSON/NPZ/images together.
- See `ros2_ws/diagnostics/README.md` for the retained datasets and cleanup record.
