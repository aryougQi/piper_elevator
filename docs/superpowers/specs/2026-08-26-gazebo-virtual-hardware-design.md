# Gazebo Virtual Hardware Design

## Status

Approved in conversation on 2026-08-26 and revised after scope clarification to
focus on the elevator button rather than a complete elevator. This document
describes the first Gazebo milestone.

## Background

The current `piper_elevator_app` simulation starts MoveIt and a standalone
`ros2_control_node` backed by `mock_components/GenericSystem`. Joint commands
are copied into simulated joint states, RViz displays the resulting transforms,
and `mock_button_pose` supplies a fixed target. There is no physics engine,
rendered camera image, depth sensor, or elevator environment.

The project needs a modern Gazebo simulation which replaces FakeSystem in the
simulation path and presents hardware-like ROS interfaces to the existing
application. The detector, planner, and motion logic must remain independent of
whether their data comes from Gazebo or from the real Piper and RealSense
hardware.

ROS 2 Humble's supported modern Gazebo pairing is Gazebo Fortress. Gazebo
Classic is outside the scope of this work.

An official-source audit was completed before implementation. AgileX's
`agx_arm_sim` does contain a Piper Gazebo launch, but it targets Gazebo Classic,
the stock two-joint gripper, and a combined Gazebo + MoveIt bringup. The whole
repository also exports an `agx_arm_description` package that conflicts with
the same-named package already downloaded through `agx_arm_ros`. It is therefore
not added wholesale. The project does not need a complete elevator model:
Gazebo's official Contact, `TouchPlugin`, and `TriggeredPublisher` systems cover
button contact and event publication, while SDFormat already defines prismatic
joint limits, damping, and spring return. Official robot and camera descriptions
are composed directly; only the small elevator-button test fixture and
Fortress-specific integration remain local.

## Goals

- Add an independent `piper_elevator_gazebo` ROS 2 package.
- Reuse official Piper, Pika, and D405 geometry instead of recreating models.
- Use Gazebo physics and `gz_ros2_control/GazeboSimSystem` for the Piper and
  Pika joints.
- Ensure the Gazebo launch path contains no
  `mock_components/GenericSystem`.
- Simulate an eye-in-hand RGB-D camera with a RealSense-compatible ROS topic
  contract.
- Simulate a deterministic elevator `up` button which can be rendered for
  YOLOv10, physically depressed by the Piper tool, spring back, and publish a
  contact event.
- Launch only virtual hardware from the Gazebo convenience script. MoveIt,
  detector, and planner remain separately launched application components.
- Preserve the current real-hardware control path for this milestone.

## Non-goals

- Removing GenericSystem from the current real Piper trajectory proxy.
- Simulating a complete elevator cabin, floor doors, floor dispatch, or people.
- Claiming real-world force/feel fidelity from the simulated button spring.
- Publishing a ground-truth button pose to the planner.
- Replacing the existing YOLO model with a simulator-only detector.
- Automatically executing motion when the simulator starts.
- Modeling RealSense optics and noise with hardware-certification accuracy.

## Selected Architecture

Simulation assets live in a separate package:

```text
ros2_ws/src/piper_elevator_gazebo/
├── config/       Gazebo ros2_control and bridge configuration
├── launch/       Virtual-hardware-only launch files
├── models/       Local elevator-button test fixture
├── piper_elevator_gazebo/
│   └── __init__.py
├── resource/
├── test/
├── urdf/         Gazebo-specific robot and sensor extensions
└── worlds/       Deterministic button-press test world
```

`piper_elevator_app` continues to own the application, MoveIt configuration,
planner, and detector. Shared robot geometry remains single-source rather than
being copied into the Gazebo package. Piper comes from the downloaded AgileX
`agx_arm_description`; Pika uses the existing package derived from the pinned
official `agx_arm_sim` model; and the camera calls RealSense's official
`sensor_d405` xacro macro.

### Robot description layering

The existing combined robot description will be separated into thin
composition layers:

1. A shared Piper + Pika + TCP composition layer which includes the official
   Piper and official-derived Pika descriptions without copying their links,
   inertials, visuals, collisions, or meshes.
2. The existing legacy control wrapper used by the current real-hardware path.
3. A Gazebo wrapper which adds `GazeboSimSystem`, the Gazebo control plugin,
   and the RGB-D sensor.

The Gazebo wrapper must be testable by expanding xacro and asserting that the
result contains `gz_ros2_control/GazeboSimSystem` and does not contain
`mock_components/GenericSystem`.

### Control ownership

Gazebo owns the simulation controller manager. The Gazebo launch must not start
a second standalone `ros2_control_node`.

```text
MoveIt FollowJointTrajectory request
                |
                v
        /arm_controller
                |
                v
          gz_ros2_control
                |
                v
        Gazebo joint physics
                |
                v
       measured joint states
```

Controller names remain stable:

- `arm_controller`
- `pika_gripper_controller`
- `joint_state_broadcaster`

The arm controller exposes
`/arm_controller/follow_joint_trajectory`, matching the action interface MoveIt
already uses. The virtual hardware layer normalizes joint feedback to
`/piper_pika/joint_states`, which is the combined feedback topic used by the
current real launch.

`piper_pika_moveit.launch.py` will gain explicit external-hardware switches so
it can start MoveIt without starting a second controller manager, controller
spawners, or duplicate robot-state publisher. Existing defaults preserve the
real launch's present behavior.

## Virtual RGB-D Camera

The official RealSense D405 description is rigidly attached to `tcp_link` with
nominal extrinsics enabled:

```text
tcp_link
└── camera_bottom_screw_frame
    └── camera_link
        ├── camera_depth_optical_frame
        └── camera_color_optical_frame
```

The D405 body mesh, collision, inertial, and optical-frame rotations come from
`realsense2_description/urdf/_d405.urdf.xacro`; they are not recreated in this
project. Only the mount translation/rotation and the Gazebo RGB-D sensor tag
are local. Defaults use the current simulation's approximate eye-in-hand
location; calibrated real extrinsics can replace them later without changing
perception code.

Initial sensor settings are:

- 848 by 480 pixels
- 30 Hz
- registered color and depth pixels
- approximately 0.1 to 2.0 metre useful depth range
- ROS optical-frame convention on `camera_color_optical_frame`

Gazebo transport topics remain internal. `ros_gz_bridge` exposes the same
external topic names used by the D405:

| External topic | ROS type | Required semantics |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/msg/Image` | BGR/RGB image convertible by `cv_bridge` |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/msg/Image` | Gazebo: `32FC1` metres; real D405: `16UC1` millimetres |
| `/camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | matching intrinsics and optical frame |

Gazebo Sim 6's RGB-D sensor publishes floating-point depth in metres and
`ros_gz_bridge` maps it to ROS `32FC1`. No conversion node is needed: the
existing detector already leaves floating-point samples in metres and applies
`depth_unit_scale: 0.001` only to integer RealSense depth. A characterization
test must lock this dual-encoding contract in place.

All simulator-side ROS nodes use `use_sim_time` and camera messages carry
Gazebo timestamps. Color, depth, and CameraInfo must be close enough for the
detector's current approximate-time synchronizer.

## Elevator Button Fixture

The deterministic world is intentionally small: floor, a short wall/panel,
lighting, Piper + Pika + camera, and one `up` button in the robot workspace. It
does not contain an elevator cabin, doors, shaft, or floor controller. This
keeps the simulator focused on the project's actual boundary: RGB-D detection,
3-D localization, approach, contact, and button travel.

No compatible Fortress model was found that provides the exact physical and
visual button contract. The button's small SDF geometry is therefore local,
but its behavior uses only official primitives: a prismatic joint with a 4 mm
travel limit and spring/damping dynamics, a contact sensor, Gazebo Sim 6
`TouchPlugin`, and `ros_gz_bridge`. No custom Gazebo plugin is required.

The first target class is `up`, which exists in the current ONNX metadata. The
panel uses raised collision geometry plus a realistic face material containing
the up symbol. Lighting, roughness, camera pose, and optional image noise are
configuration points because the existing model was trained on real images and
may exhibit a synthetic-to-real domain gap.

No world plugin publishes button ground truth to `/button_pose`. Ground-truth
model coordinates may be read only by integration tests to compare perception
error. If YOLO cannot recognize the rendered target, the response is to improve
the scene assets or collect synthetic training images, not to substitute
`mock_button_pose`.

The button face moves only along its panel normal and returns when released.
`TouchPlugin` publishes Gazebo Boolean state on `/elevator_button/touched`; the
official bridge exposes it as ROS `std_msgs/msg/Bool` on
`/elevator_button/pressed`. Joint state is retained as test instrumentation so
tests can distinguish merely touching the face from depressing it through the
configured threshold. The official plugin is one-shot, so repeatable tests
re-arm it through `/elevator_button/enable` before each attempt.
`TriggeredPublisher` is optional for later LED or mission events and does not
justify a custom plugin.

## Launch Boundary

The convenience command is intentionally hardware-only:

```bash
./scripts/gazebo_hardware.sh
```

It starts:

- Gazebo server and optional GUI
- the elevator-button test world
- Piper + Pika + RGB-D entity spawning
- Gazebo's controller manager and controller spawners
- the virtual camera bridges
- joint-state normalization
- `/clock`
- robot-state publication when required by the spawn/control integration

It does **not** start:

- MoveIt or RViz
- `button_detector`
- `button_approach_planner`
- `mock_button_pose`
- automatic planning or execution

Application components are launched separately, just as they are when bringing
up real hardware:

```bash
# Planning infrastructure connected to an already-running controller manager.
ros2 launch piper_elevator_app piper_pika_moveit.launch.py \
  external_hardware:=true use_sim_time:=true

# Existing perception core; the input topic names are unchanged.
ros2 launch piper_elevator_app button_detector.launch.py \
  use_sim_time:=true

# Existing planning core.
ros2 launch piper_elevator_app button_approach_planner.launch.py \
  use_sim_time:=true
```

The exact final argument names will be fixed in the implementation plan and
documented in `develop.md`; the important contract is that the virtual-hardware
script never starts application-core nodes.

## Simulation and Real-Hardware Parity

Application source code must not branch on `gazebo` versus `real`. Differences
are limited to launch-time hardware ownership, time source, and calibrated
extrinsics.

| Boundary | Gazebo provider | Real provider | Application contract |
|---|---|---|---|
| Color | Gazebo RGB-D sensor | RealSense driver | same image topic/type |
| Aligned depth | Gazebo bridge, `32FC1` metres | RealSense driver, `16UC1` millimetres | same topic; existing detector estimates metres from either encoding |
| Intrinsics | Gazebo bridge | RealSense driver | same CameraInfo topic |
| Arm trajectory | Gazebo arm controller | current MoveIt/controller proxy | same FollowJointTrajectory action |
| Joint feedback | Gazebo normalization | Piper/Pika mux | `/piper_pika/joint_states` |
| Button pose | existing YOLO detector | existing YOLO detector | `/button_pose` in camera optical frame |

`/elevator_button/pressed` is simulator test instrumentation, not an interface
that application code may require from the real elevator.

Moving to the real robot still requires selecting the real hardware launch,
using real camera extrinsics, enforcing the existing control gate, and starting
at a safe speed. Matching ROS topic names does not replace hand-eye calibration
or real-hardware safety validation.

## Failure Behaviour

- A missing Gazebo control plugin or controller-manager timeout stops controller
  startup with a visible error; no FakeSystem fallback is allowed.
- Missing camera bridges or invalid message encodings prevent perception startup
  validation rather than silently changing topic names.
- The detector retains its current strict CUDA behavior. CPU execution is only
  used when explicitly requested for debugging.
- No selected/detected button means no `/button_pose`; therefore the planner has
  no valid target and cannot execute a button approach.
- Contact without sufficient prismatic travel fails the button-press acceptance
  check instead of being reported as a successful press.
- Automatic execution remains disabled. Gazebo hardware startup alone never
  moves the arm.
- A failed YOLO synthetic-image test is reported as an unmet visual acceptance
  criterion, not hidden behind ground-truth output.

## Dependencies and Container Changes

The Humble container adds the binary packages required for modern Gazebo ROS
integration, principally `ros-humble-ros-gz` and
`ros-humble-gz-ros2-control`. Existing host networking, X11, GPU, and workspace
mount configuration is reused. The launcher supports `gui:=false` for headless
smoke tests.

The new package declares only its direct runtime and test dependencies. It
depends on `piper_elevator_app` for the shared robot description and controller
contract but must not import detector or planner implementation code.

## Verification Strategy

Verification is layered so failures identify the responsible boundary.

1. **Static description checks**
   - Expand the Gazebo xacro.
   - Parse the generated XML.
   - Confirm GazeboSimSystem is present and GenericSystem is absent.
   - Confirm camera frames, joint names, and required plugins exist once.
2. **Depth encoding characterization test**
   - Feed equivalent `float32` metre and `uint16` millimetre depth images to
     the existing detector depth estimator.
   - Require both encodings to produce the same metre-valued result.
3. **Headless launch smoke test**
   - Start the world and robot with `gui:=false`.
   - Confirm `/clock`, controller manager, normalized joint states, and all three
     external camera topics appear.
4. **Controller integration test**
   - Confirm the three expected controllers are active.
   - Send a small FollowJointTrajectory goal.
   - Observe physical joint feedback converge within tolerance.
   - Confirm no standalone FakeSystem controller manager is running.
5. **Camera contract test**
   - Verify resolution, approximate frame rate, timestamps, frame IDs, and
     Gazebo-native `32FC1` depth encoding in metres.
6. **Button mechanics test**
   - Push the face along the panel normal, observe up to 4 mm joint travel, and
     require it to return after release.
   - Require `/elevator_button/pressed` only when the Piper tool contacts the
     button; touching the surrounding panel must not trigger it.
7. **Perception acceptance test**
   - Start the unchanged detector and select class `up`.
   - Require stable `/button_detection_valid` and `/button_pose` output.
   - Compare the detected 3-D target to test-only world ground truth with a
     documented tolerance.
8. **Planning acceptance test**
   - Start MoveIt and the existing planner separately.
   - Plan, then explicitly execute, an approach trajectory.
   - Verify the TCP stops at the configured pre-contact offset without requiring
     any mock button-pose publisher.

## Acceptance Criteria

The milestone is complete when:

- `./scripts/gazebo_hardware.sh` starts only the virtual hardware boundary.
- The expanded Gazebo robot contains GazeboSimSystem and no GenericSystem.
- Existing detector and planner executables run without simulation-specific
  source-code branches.
- Virtual camera topics match the current RealSense names, and the unchanged
  detector produces metre-valued depth from Gazebo `32FC1` and real `16UC1`.
- The current YOLO model detects the rendered `up` target and produces a stable
  3-D pose. If model adaptation or additional training is required, this
  milestone remains incomplete until that work passes the same test.
- The button physically travels, returns after release, and publishes the
  bridged press event without any custom Gazebo plugin.
- MoveIt can execute a planned approach through the Gazebo controller.
- The existing real launch still behaves as it did before this work.
- `develop.md` contains concise commands for virtual hardware startup, topic
  inspection, detector startup, planning, and execution.

## References

- AgileX official arm simulation repository and Gazebo Classic integration:
  <https://github.com/agilexrobotics/agx_arm_sim>
- Intel RealSense official ROS description repository:
  <https://github.com/IntelRealSense/realsense-ros/tree/ros2-master/realsense2_description>
- Gazebo ROS 2 installation and version pairing:
  <https://gazebosim.org/docs/jetty/ros_installation/>
- `gz_ros2_control` for ROS 2 Humble:
  <https://control.ros.org/humble/doc/gz_ros2_control/doc/index.html>
- Gazebo ROS 2 interoperability:
  <https://gazebosim.org/docs/fortress/ros2_interop/>
- Official `ros_gz_bridge` RGB-D bridge example:
  <https://github.com/gazebosim/ros_gz/blob/ros2/ros_gz_sim_demos/config/rgbd_camera_bridge.yaml>
- Gazebo Sim 6 built-in contact and message systems:
  <https://gazebosim.org/api/sim/6/classignition_1_1gazebo_1_1systems_1_1TouchPlugin.html>
  and
  <https://gazebosim.org/api/sim/6/classignition_1_1gazebo_1_1systems_1_1TriggeredPublisher.html>
- Gazebo Sim 6 built-in system inventory, including Contact,
  JointStatePublisher, and ApplyLinkWrench:
  <https://gazebosim.org/api/sim/6/namespaceignition_1_1gazebo_1_1systems.html>
- SDFormat joint limits, damping, and spring dynamics:
  <https://sdformat.org/spec/1.8/joint/>
- Official Gazebo contact / TouchPlugin tutorial source:
  <https://github.com/gazebosim/docs/blob/master/dome/sensors.md>
