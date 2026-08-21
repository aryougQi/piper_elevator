# Pika gripper description

This package provides the real AgileX Pika visual geometry used by the Piper
MoveIt simulation. The model comes from AgileX's official `agx_arm_sim`
repository; see `NOTICE` for the exact commit and source files.

The visual geometry is unchanged. The two finger joints are driven from one
`center_joint` mimic model so the simulated opening uses the same total opening
distance (0 to 0.098 m) published by the official `sensor_tools` driver.
