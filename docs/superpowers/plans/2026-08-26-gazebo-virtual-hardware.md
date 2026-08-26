# Gazebo Virtual Hardware Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Gazebo Fortress virtual-hardware package for Piper + Pika + an eye-in-hand RGB-D camera, with hardware-like ROS interfaces and no FakeSystem in the Gazebo path.

**Architecture:** `piper_elevator_gazebo` is a thin integration package around official robot descriptions and official ROS/Gazebo plugins. It owns only the project-specific Gazebo extensions, elevator world, bridge configuration, depth normalization, and hardware-only launch script. `piper_elevator_app` keeps MoveIt, detector, and planner code, and gains a description-only robot model plus an external-hardware launch mode so those components can connect to Gazebo without starting a second controller manager.

**Tech Stack:** ROS 2 Humble, Gazebo Fortress, `ros_gz_sim`, `ros_gz_bridge`, `gz_ros2_control`, xacro/URDF, SDFormat, `ros2_control`, Python 3, `rclpy`, NumPy, pytest, Docker Compose.

## Global Constraints

- Use ROS 2 Humble with Gazebo Fortress; do not add Gazebo Classic.
- The expanded Gazebo robot must contain `gz_ros2_control/GazeboSimSystem` and must not contain `mock_components/GenericSystem`.
- `./scripts/gazebo_hardware.sh` starts virtual hardware only: no MoveIt, RViz, detector, planner, mock button pose, planning, or execution.
- Keep the current real-hardware trajectory proxy unchanged in this milestone.
- Preserve the external camera contract exactly: `/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`, and `/camera/color/camera_info`.
- Normalize simulated depth to `sensor_msgs/msg/Image` encoding `16UC1` in millimetres so `depth_unit_scale: 0.001` remains unchanged.
- Normalize simulated Piper + Pika feedback to `/piper_pika/joint_states` and keep `/arm_controller/follow_joint_trajectory` as the MoveIt execution action.
- Do not publish Gazebo ground truth to `/button_pose`; the existing YOLO inference path must produce that message.
- Keep automatic motion disabled. Planning and execution are explicit user actions.
- Preserve unrelated user work, especially the existing deletion of `AI_PROJECT_CONTEXT.md`; never stage that path.

## Official Reuse Audit

Checked against AgileX `agx_arm_sim` commit
`f8cd8b147c75d59e14f90fb0646770eefa268ed0`, the currently downloaded
`agx_arm_ros` ROS 2 package, the currently downloaded RealSense description,
and the ROS 2 Humble Gazebo packages.

Use these official components directly:

- Piper links, joints, inertials, collisions, and meshes from the downloaded
  `agx_arm_description` package. Never copy or redraw the Piper model.
- Pika geometry from AgileX `agx_arm_sim`. The existing
  `pika_gripper_description` package already pins those official meshes and
  dimensions; its only local adaptation is the `center_joint` interface used by
  the real Pika ROS driver.
- D405 body, collision, inertial, mesh, and complete nominal frame tree from
  `realsense2_description/urdf/_d405.urdf.xacro` by calling the official
  `sensor_d405` macro. Do not create a camera box, mesh, or optical-frame chain.
- Gazebo Fortress process and entity creation from `ros_gz_sim`, transport
  conversion from `ros_gz_bridge`, simulation control from `gz_ros2_control`,
  controllers from `ros2_controllers`, and joint-state forwarding from
  `topic_tools relay`. Do not implement replacements for these nodes/plugins.

Do **not** add the whole `agx_arm_sim` repository to this workspace. Its
`agx_arm_description` package name conflicts with the package already supplied
by `agx_arm_ros`. Its ready-made Gazebo launch uses Gazebo Classic
(`gazebo_ros` plus `gazebo_ros2_control`), the stock two-joint Piper gripper,
and starts MoveIt in the same launch. It therefore cannot satisfy the approved
Fortress, Pika `center_joint`, and hardware-only boundary without adaptation.

The local code left in this plan is integration that no official package can
know about: the Piper/Pika/TCP composition wrapper, Fortress-specific sensor
and control tags, project topic/controller configuration, the elevator scene,
and the `32FC1` metres to RealSense-style `16UC1` millimetres boundary adapter.
The official `depth_image_proc/convert_metric_node` performs the opposite
conversion and cannot replace that adapter.

---

## File Map

### New package

- `ros2_ws/src/piper_elevator_gazebo/package.xml` — direct ROS/Gazebo dependencies.
- `ros2_ws/src/piper_elevator_gazebo/setup.py` — Python entry points and recursive asset installation.
- `ros2_ws/src/piper_elevator_gazebo/setup.cfg` — ROS executable install location.
- `ros2_ws/src/piper_elevator_gazebo/resource/piper_elevator_gazebo` — ament resource marker.
- `ros2_ws/src/piper_elevator_gazebo/piper_elevator_gazebo/depth_conversion.py` — pure NumPy depth conversion.
- `ros2_ws/src/piper_elevator_gazebo/piper_elevator_gazebo/depth_adapter.py` — ROS image adapter node.
- `ros2_ws/src/piper_elevator_gazebo/config/gazebo_controllers.yaml` — Gazebo controller manager configuration.
- `ros2_ws/src/piper_elevator_gazebo/config/camera_bridge.yaml` — Gazebo-to-ROS camera and clock bridge contract.
- `ros2_ws/src/piper_elevator_gazebo/urdf/piper_pika_gazebo.urdf.xacro` — Gazebo control and RGB-D extension.
- `ros2_ws/src/piper_elevator_gazebo/models/elevator_panel/model.config` — Gazebo model metadata.
- `ros2_ws/src/piper_elevator_gazebo/models/elevator_panel/model.sdf` — panel collision and visuals.
- `ros2_ws/src/piper_elevator_gazebo/models/elevator_panel/meshes/up_symbol.dae` — code-native up-arrow face mesh.
- `ros2_ws/src/piper_elevator_gazebo/worlds/elevator.sdf` — deterministic elevator test world.
- `ros2_ws/src/piper_elevator_gazebo/launch/gazebo_hardware.launch.py` — virtual-hardware-only bringup.
- `ros2_ws/src/piper_elevator_gazebo/test/test_depth_conversion.py` — conversion edge cases.
- `ros2_ws/src/piper_elevator_gazebo/test/test_gazebo_description.py` — xacro and SDF contract tests.
- `ros2_ws/src/piper_elevator_gazebo/test/test_launch_boundary.py` — asserts hardware launch excludes core nodes.
- `ros2_ws/src/piper_elevator_gazebo/test/virtual_hardware_probe.py` — running-system controller and sensor probe.
- `ros2_ws/src/piper_elevator_gazebo/test/perception_probe.py` — YOLO and 3-D pose acceptance probe.

### Existing application and infrastructure

- `ros2_ws/src/piper_elevator_app/config/piper_pika_description.xacro` — thin composition wrapper for official Piper, official-derived Pika, and project TCP.
- `ros2_ws/src/piper_elevator_app/config/piper_pika_description.urdf.xacro` — description-only MoveIt model.
- `ros2_ws/src/piper_elevator_app/config/piper_pika.urdf.xacro` — legacy GenericSystem wrapper, unchanged by default.
- `ros2_ws/src/piper_elevator_app/piper_elevator_app/launch_mode.py` — pure external/legacy MoveIt launch policy.
- `ros2_ws/src/piper_elevator_app/launch/piper_pika_moveit.launch.py` — external-hardware and simulated-time switches.
- `ros2_ws/src/piper_elevator_app/launch/button_detector.launch.py` — `use_sim_time` pass-through.
- `ros2_ws/src/piper_elevator_app/launch/button_approach_planner.launch.py` — `use_sim_time` pass-through.
- `ros2_ws/src/piper_elevator_app/test/test_robot_description.py` — shared/legacy description regression tests.
- `ros2_ws/src/piper_elevator_app/test/test_launch_mode.py` — launch policy unit tests.
- `docker/Dockerfile` — Fortress ROS integration packages.
- `scripts/gazebo_hardware.sh` — only supported virtual-hardware convenience command.
- `scripts/check_gazebo.sh` — headless virtual-hardware smoke test.
- `scripts/check_gazebo_perception.sh` — detector acceptance test.
- `scripts/check_gazebo_approach.sh` — separately launched application integration test.
- `scripts/sim.sh` — redirect the generic simulation command to Gazebo hardware.
- `scripts/check.sh` — package and launch regression checks.
- `develop.md` — concise operator commands, preserving existing content.
- `README.md` — architecture, dependencies, and migration guidance.

---

### Task 1: Install Gazebo dependencies and scaffold the simulation package

**Files:**
- Modify: `docker/Dockerfile`
- Create: `ros2_ws/src/piper_elevator_gazebo/package.xml`
- Create: `ros2_ws/src/piper_elevator_gazebo/setup.py`
- Create: `ros2_ws/src/piper_elevator_gazebo/setup.cfg`
- Create: `ros2_ws/src/piper_elevator_gazebo/resource/piper_elevator_gazebo`
- Create: `ros2_ws/src/piper_elevator_gazebo/piper_elevator_gazebo/__init__.py`
- Modify: `scripts/check.sh`

**Interfaces:**
- Consumes: the existing `piper_ros2` Docker service and ROS 2 workspace layout.
- Produces: an installable `piper_elevator_gazebo` ament package and the Fortress binaries used by every later task.

- [ ] **Step 1: Add a failing package preflight**

Add `piper_elevator_gazebo` to `expected_packages` in `scripts/check.sh`, change the final count from 13 to 14, and add:

```bash
ros2 pkg prefix ros_gz_sim >/dev/null
ros2 pkg prefix ros_gz_bridge >/dev/null
ros2 pkg prefix gz_ros2_control >/dev/null
```

- [ ] **Step 2: Run the preflight to verify the package is absent**

Run `./scripts/check.sh`.

Expected: FAIL at `ros2 pkg prefix piper_elevator_gazebo` or a Gazebo package lookup.

- [ ] **Step 3: Add binary dependencies to the image**

Add beside the other ROS packages in `docker/Dockerfile`:

```dockerfile
        ros-humble-gz-ros2-control \
        ros-humble-ros-gz \
```

Do not install `gazebo_ros_pkgs` or `gazebo_ros2_control`.

- [ ] **Step 4: Create the package manifest**

Use package format 3 with `ament_python`. Declare direct runtime dependencies on `ament_index_python`, `controller_manager`, `cv_bridge`, `gz_ros2_control`, `joint_state_broadcaster`, `joint_trajectory_controller`, `launch`, `launch_ros`, `piper_elevator_app`, `python3-numpy`, `rclpy`, `realsense2_description`, `robot_state_publisher`, `ros_gz_bridge`, `ros_gz_sim`, `rosgraph_msgs`, `sensor_msgs`, `topic_tools`, and `xacro`. Declare test dependencies on `ament_flake8`, `ament_pep257`, `control_msgs`, `python3-pytest`, and `tf2_ros`.

Export the build type and model path:

```xml
<export>
  <build_type>ament_python</build_type>
  <gazebo_ros gazebo_model_path="${prefix}/models"/>
</export>
```

- [ ] **Step 5: Create recursive asset installation**

Use this helper in `setup.py` so nested assets keep their directories:

```python
from pathlib import Path
from setuptools import find_packages, setup


PACKAGE_NAME = 'piper_elevator_gazebo'


def asset_data_files(root_name):
    grouped = {}
    for path in Path(root_name).rglob('*'):
        if path.is_file():
            destination = Path('share') / PACKAGE_NAME / path.parent
            grouped.setdefault(str(destination), []).append(str(path))
    return sorted(grouped.items())
```

Install `config`, `launch`, `models`, `urdf`, and `worlds`. Define `depth_adapter = piper_elevator_gazebo.depth_adapter:main` as the console entry point.

- [ ] **Step 6: Build the image and workspace**

```bash
docker compose build piper_ros2
./scripts/build.sh
```

Expected: the image installs Fortress packages and colcon installs `piper_elevator_gazebo`.

- [ ] **Step 7: Verify installed prefixes**

```bash
docker compose run --rm piper_ros2 bash -lc '
  source /workspace/ros2_ws/install/setup.bash
  ros2 pkg prefix piper_elevator_gazebo
  ros2 pkg prefix ros_gz_sim
  ros2 pkg prefix ros_gz_bridge
  ros2 pkg prefix gz_ros2_control
'
```

Expected: four prefixes and exit code 0.

- [ ] **Step 8: Commit**

```bash
git add docker/Dockerfile scripts/check.sh ros2_ws/src/piper_elevator_gazebo
git commit -m "build: add Gazebo virtual hardware package"
```

---

### Task 2: Separate shared geometry from the legacy FakeSystem wrapper

**Files:**
- Create: `ros2_ws/src/piper_elevator_app/config/piper_pika_description.xacro`
- Create: `ros2_ws/src/piper_elevator_app/config/piper_pika_description.urdf.xacro`
- Modify: `ros2_ws/src/piper_elevator_app/config/piper_pika.urdf.xacro`
- Create: `ros2_ws/src/piper_elevator_app/test/test_robot_description.py`

**Interfaces:**
- Consumes: official Piper/Pika descriptions, existing TCP arguments, and initial-position YAML.
- Produces: a description-only robot with no control element and a backwards-compatible legacy wrapper with `PiperPikaFakeSystem`.

- [ ] **Step 1: Write failing description tests**

```python
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE_ROOT / 'config'


def expand(filename):
    command = ['xacro', str(CONFIG / filename)]
    if filename == 'piper_pika.urdf.xacro':
        command.append(
            f'initial_positions_file:={CONFIG / "piper_pika_initial_positions.yaml"}'
        )
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True
    )
    return ET.fromstring(completed.stdout)


def test_description_only_robot_has_no_ros2_control():
    robot = expand('piper_pika_description.urdf.xacro')
    assert robot.find('ros2_control') is None
    assert robot.find("link[@name='tcp_link']") is not None


def test_legacy_wrapper_keeps_fake_system_contract():
    robot = expand('piper_pika.urdf.xacro')
    control = robot.find("ros2_control[@name='PiperPikaFakeSystem']")
    assert control.findtext('hardware/plugin') == 'mock_components/GenericSystem'
    assert [joint.get('name') for joint in control.findall('joint')] == [
        'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
        'center_joint',
    ]
```

- [ ] **Step 2: Verify the new file is missing**

Run the focused pytest inside the container. Expected: FAIL because `piper_pika_description.urdf.xacro` is absent.

- [ ] **Step 3: Extract the shared geometry fragment**

Move only the composition calls, Pika mount, and TCP link/joint into
`piper_pika_description.xacro`. Continue including the official Piper macro and
the existing official-derived Pika macro; do not copy their link, inertial,
collision, or mesh definitions into this fragment. Expose:

```xml
<xacro:macro name="piper_pika_end_effector"
             params="tcp_offset_xyz tcp_offset_rpy">
  <xacro:pika_gripper parent="link6"/>
  <link name="tcp_link"/>
  <joint name="tcp_joint" type="fixed">
    <origin xyz="${tcp_offset_xyz}" rpy="${tcp_offset_rpy}"/>
    <parent link="pika_gripper_base_link"/>
    <child link="tcp_link"/>
  </joint>
</xacro:macro>
```

- [ ] **Step 4: Add the description-only wrapper**

Declare the existing TCP arguments, include the fragment, and call `piper_pika_end_effector`. Do not add `<ros2_control>`.

- [ ] **Step 5: Rebuild the legacy wrapper**

Keep its arguments, initial positions, seven joint interfaces, control name, and GenericSystem plugin equivalent. Replace only duplicated geometry with the shared fragment.

- [ ] **Step 6: Run tests and legacy smoke checks**

```bash
docker compose run --rm piper_ros2 bash -lc '
  source /workspace/ros2_ws/install/setup.bash
  cd /workspace/ros2_ws/src/piper_elevator_app
  pytest -q test/test_robot_description.py test/test_motion_core.py test/test_button_detector.py
'
./scripts/build.sh
./scripts/check.sh
```

Expected: all tests pass and legacy controllers reach ready state.

- [ ] **Step 7: Commit**

```bash
git add ros2_ws/src/piper_elevator_app/config ros2_ws/src/piper_elevator_app/test/test_robot_description.py
git commit -m "refactor: separate Piper Pika robot description"
```

---

### Task 3: Implement the missing RealSense-compatible depth boundary adapter

**Files:**
- Create: `ros2_ws/src/piper_elevator_gazebo/piper_elevator_gazebo/depth_conversion.py`
- Create: `ros2_ws/src/piper_elevator_gazebo/piper_elevator_gazebo/depth_adapter.py`
- Create: `ros2_ws/src/piper_elevator_gazebo/test/test_depth_conversion.py`

**Interfaces:**
- Consumes: `32FC1` metres on `/sim/camera/depth_image`.
- Produces: `16UC1` millimetres on `/camera/aligned_depth_to_color/image_raw`, preserving the header.

This custom node exists only because the official Gazebo bridge preserves the
floating-point depth representation, while the official
`depth_image_proc/convert_metric_node` converts `16UC1` millimetres to `32FC1`
metres, not the required reverse direction. Do not add any color, CameraInfo,
TF, or generic bridge behavior to this adapter.

- [ ] **Step 1: Write failing conversion tests**

```python
import numpy as np
import pytest

from piper_elevator_gazebo.depth_conversion import metres_to_uint16_mm


def test_converts_valid_samples():
    source = np.array([[0.1, 0.4564, 2.0]], dtype=np.float32)
    result = metres_to_uint16_mm(source, 0.1, 2.0)
    np.testing.assert_array_equal(
        result, np.array([[100, 456, 2000]], dtype=np.uint16)
    )


def test_zeros_invalid_samples():
    source = np.array(
        [[np.nan, np.inf, -0.1, 0.099, 2.001]], dtype=np.float32
    )
    result = metres_to_uint16_mm(source, 0.1, 2.0)
    np.testing.assert_array_equal(result, np.zeros((1, 5), dtype=np.uint16))


def test_rejects_bad_limits():
    with pytest.raises(ValueError, match='0 < min_depth_m < max_depth_m'):
        metres_to_uint16_mm(np.ones((1, 1), dtype=np.float32), 2.0, 0.1)
```

- [ ] **Step 2: Verify import failure**

Run `pytest -q test/test_depth_conversion.py`. Expected: FAIL because the module is absent.

- [ ] **Step 3: Implement the converter**

```python
def metres_to_uint16_mm(depth_metres, min_depth_m, max_depth_m):
    if not 0.0 < min_depth_m < max_depth_m:
        raise ValueError('expected 0 < min_depth_m < max_depth_m')
    source = np.asarray(depth_metres, dtype=np.float32)
    result = np.zeros(source.shape, dtype=np.uint16)
    valid = (
        np.isfinite(source)
        & (source >= min_depth_m)
        & (source <= max_depth_m)
    )
    millimetres = np.rint(source[valid] * 1000.0)
    result[valid] = np.clip(millimetres, 0, 65535).astype(np.uint16)
    return result
```

- [ ] **Step 4: Implement the ROS adapter**

Declare `input_topic=/sim/camera/depth_image`, `output_topic=/camera/aligned_depth_to_color/image_raw`, `min_depth_m=0.1`, and `max_depth_m=2.0`. Use best-effort sensor QoS depth 1. Reject non-`32FC1` input, convert with the pure function, publish `16UC1`, and copy the source header exactly.

- [ ] **Step 5: Run unit and style tests**

```bash
docker compose run --rm piper_ros2 bash -lc '
  source /workspace/ros2_ws/install/setup.bash
  cd /workspace/ros2_ws/src/piper_elevator_gazebo
  pytest -q test/test_depth_conversion.py
  ament_flake8 piper_elevator_gazebo test/test_depth_conversion.py
  ament_pep257 piper_elevator_gazebo
'
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add ros2_ws/src/piper_elevator_gazebo
git commit -m "feat: normalize simulated depth for detector"
```

---

### Task 4: Add Gazebo control and RGB-D robot description

**Files:**
- Create: `ros2_ws/src/piper_elevator_gazebo/config/gazebo_controllers.yaml`
- Create: `ros2_ws/src/piper_elevator_gazebo/urdf/piper_pika_gazebo.urdf.xacro`
- Create: `ros2_ws/src/piper_elevator_gazebo/test/test_gazebo_description.py`

**Interfaces:**
- Consumes: official Piper geometry, official-derived Pika geometry, the
  official RealSense D405 xacro, shared composition, and initial positions.
- Produces: seven physical position-command joints and `/piper_d405/{image,depth_image,camera_info}` Gazebo topics.

- [ ] **Step 1: Write the failing xacro contract test**

Expand the Gazebo xacro and assert:

```python
assert robot.findtext('ros2_control/hardware/plugin') == (
    'gz_ros2_control/GazeboSimSystem'
)
assert 'mock_components/GenericSystem' not in expanded_xml
assert expanded_xml.count('gz_ros2_control::GazeboSimROS2ControlPlugin') == 1
assert robot.find("link[@name='camera_bottom_screw_frame']") is not None
assert robot.find("link[@name='camera_link']") is not None
assert robot.find("link[@name='camera_color_optical_frame']") is not None
assert robot.find(
    "gazebo[@reference='camera_link']/sensor[@type='rgbd_camera']"
) is not None
```

Also require ros2_control joints `joint1` through `joint6` and `center_joint` in that order.

- [ ] **Step 2: Confirm the Gazebo xacro is missing**

Run the focused pytest. Expected: FAIL on missing file.

- [ ] **Step 3: Add controller YAML**

Use a 200 Hz manager update rate. Keep controller names and joint lists identical to `piper_pika_ros2_controllers.yaml`, with position commands, position/velocity state, and nonzero end velocity disabled.

- [ ] **Step 4: Add GazeboSimSystem and plugin**

```xml
<ros2_control name="PiperPikaGazeboSystem" type="system">
  <hardware>
    <plugin>gz_ros2_control/GazeboSimSystem</plugin>
  </hardware>
  <!-- seven tested joint interface blocks -->
</ros2_control>
<gazebo>
  <plugin filename="gz_ros2_control-system"
          name="gz_ros2_control::GazeboSimROS2ControlPlugin">
    <parameters>$(arg controllers_file)</parameters>
    <controller_manager_name>controller_manager</controller_manager_name>
  </plugin>
</gazebo>
```

- [ ] **Step 5: Compose the official D405 and add only the Gazebo sensor tag**

Declare `camera_xyz` default `0 0 -0.10` and `camera_rpy` default `0 0 0`.
Include the D405 description already shipped in the downloaded official
RealSense package and instantiate its complete nominal frame tree:

```xml
<xacro:include
  filename="$(find realsense2_description)/urdf/_d405.urdf.xacro"/>
<xacro:sensor_d405 parent="tcp_link" name="camera"
                   use_nominal_extrinsics="true">
  <origin xyz="$(arg camera_xyz)" rpy="$(arg camera_rpy)"/>
</xacro:sensor_d405>
```

Do not define `camera_link`, D405 dimensions, inertials, collision, mesh, or
optical joints locally. Attach only the simulator-specific sensor extension to
the official `camera_link`:

```xml
<gazebo reference="camera_link">
<sensor name="piper_d405" type="rgbd_camera">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <topic>/piper_d405</topic>
  <camera>
    <horizontal_fov>1.518436</horizontal_fov>
    <image><width>848</width><height>480</height><format>R8G8B8</format></image>
    <clip><near>0.1</near><far>2.0</far></clip>
  </camera>
</sensor>
</gazebo>
```

- [ ] **Step 6: Build and test**

```bash
./scripts/build.sh
docker compose run --rm piper_ros2 bash -lc '
  source /workspace/ros2_ws/install/setup.bash
  cd /workspace/ros2_ws/src/piper_elevator_gazebo
  pytest -q test/test_gazebo_description.py
'
```

Expected: PASS with no GenericSystem string.

- [ ] **Step 7: Commit**

```bash
git add ros2_ws/src/piper_elevator_gazebo/config ros2_ws/src/piper_elevator_gazebo/urdf ros2_ws/src/piper_elevator_gazebo/test/test_gazebo_description.py
git commit -m "feat: add Gazebo Piper Pika hardware description"
```

---

### Task 5: Build the elevator world and detectable panel

**Files:**
- Create: `ros2_ws/src/piper_elevator_gazebo/models/elevator_panel/model.config`
- Create: `ros2_ws/src/piper_elevator_gazebo/models/elevator_panel/model.sdf`
- Create: `ros2_ws/src/piper_elevator_gazebo/models/elevator_panel/meshes/up_symbol.dae`
- Create: `ros2_ws/src/piper_elevator_gazebo/worlds/elevator.sdf`
- Extend: `ros2_ws/src/piper_elevator_gazebo/test/test_gazebo_description.py`

**Interfaces:**
- Consumes: SDF 1.8 primitives and installed Gazebo resource paths.
- Produces: a lit `elevator` world whose up-button centre is `(0.55, 0.00, 0.30)` in `world`, with no ground-truth publisher.

- [ ] **Step 1: Write failing SDF tests**

Parse both SDF files and assert:

```python
assert world.get('name') == 'elevator'
assert world.find("plugin[@name='gz::sim::systems::Physics']") is not None
assert world.find("plugin[@name='gz::sim::systems::Sensors']") is not None
assert world.find("include[uri='model://elevator_panel']") is not None
assert panel.find("link/visual[@name='up_button_face']") is not None
assert panel.find("link/collision[@name='up_button_collision']") is not None
assert '/button_pose' not in panel_xml
```

- [ ] **Step 2: Verify asset failure**

Run the focused test. Expected: FAIL on missing world/model files.

- [ ] **Step 3: Create the panel**

Use a static 0.18 m by 0.10 m by 0.015 m brushed-metal box, a circular upper button with a collision cylinder, and a separate dark up-arrow face mesh. Put the model origin at the up-button centre. Use PBR roughness 0.32 for the panel and 0.18 for the rim; keep emissive intensity below 0.15.

- [ ] **Step 4: Create the up-symbol mesh**

Create Collada 1.4.1 metre-scale geometry with a centred 28 mm arrow head and 9 mm stem, 0.6 mm above the circular face, using a near-black material. Keep it symmetric and use no text label so the intended ONNX class is `up`.

- [ ] **Step 5: Create the world**

Use SDF 1.8 with Physics, UserCommands, SceneBroadcaster, and Sensors systems. Configure `ogre2`, 1000 Hz physics, gravity `0 0 -9.81`, ambient `0.35 0.35 0.35 1`, a floor, vertical wall, and directional panel light. Include:

```xml
<include>
  <uri>model://elevator_panel</uri>
  <name>elevator_panel</name>
  <pose>0.55 0 0.30 0 0 3.14159265359</pose>
</include>
```

- [ ] **Step 6: Build and test resource resolution**

```bash
./scripts/build.sh
docker compose run --rm piper_ros2 bash -lc '
  source /workspace/ros2_ws/install/setup.bash
  cd /workspace/ros2_ws/src/piper_elevator_gazebo
  pytest -q test/test_gazebo_description.py
  test -n "${GZ_SIM_RESOURCE_PATH:-${IGN_GAZEBO_RESOURCE_PATH:-}}"
'
```

Expected: PASS and a nonempty resource path.

- [ ] **Step 7: Commit**

```bash
git add ros2_ws/src/piper_elevator_gazebo/models ros2_ws/src/piper_elevator_gazebo/worlds ros2_ws/src/piper_elevator_gazebo/test/test_gazebo_description.py
git commit -m "feat: add elevator button simulation world"
```

---

### Task 6: Add hardware-only Gazebo bringup

**Files:**
- Create: `ros2_ws/src/piper_elevator_gazebo/config/camera_bridge.yaml`
- Create: `ros2_ws/src/piper_elevator_gazebo/launch/gazebo_hardware.launch.py`
- Create: `ros2_ws/src/piper_elevator_gazebo/test/test_launch_boundary.py`
- Create: `scripts/gazebo_hardware.sh`

**Interfaces:**
- Consumes: world, robot xacro, controller YAML, bridge, adapter, and `topic_tools relay`.
- Produces: `/clock`, controller manager/action, normalized joint feedback, and three RealSense-compatible camera topics.

- [ ] **Step 1: Write the failing launch-boundary test**

Read launch and script source. Require `ros_gz_sim`, `ros_gz_bridge`, `robot_state_publisher`, `depth_adapter`, and `relay`. Reject `move_group`, `rviz2`, `button_detector`, `button_approach_planner`, and `mock_button_pose` executable declarations.

- [ ] **Step 2: Verify launch failure**

Run the focused test. Expected: FAIL because launch/script paths are absent.

- [ ] **Step 3: Configure the official camera and clock bridge**

Use the official `ros_gz_bridge` `parameter_bridge` executable and its YAML
schema; this file contains only project topic mappings, not a custom bridge
node:

```yaml
- ros_topic_name: /camera/color/image_raw
  gz_topic_name: /piper_d405/image
  ros_type_name: sensor_msgs/msg/Image
  gz_type_name: gz.msgs.Image
  direction: GZ_TO_ROS
  lazy: true
- ros_topic_name: /sim/camera/depth_image
  gz_topic_name: /piper_d405/depth_image
  ros_type_name: sensor_msgs/msg/Image
  gz_type_name: gz.msgs.Image
  direction: GZ_TO_ROS
  lazy: true
- ros_topic_name: /camera/color/camera_info
  gz_topic_name: /piper_d405/camera_info
  ros_type_name: sensor_msgs/msg/CameraInfo
  gz_type_name: gz.msgs.CameraInfo
  direction: GZ_TO_ROS
  lazy: true
- ros_topic_name: /clock
  gz_topic_name: /clock
  ros_type_name: rosgraph_msgs/msg/Clock
  gz_type_name: gz.msgs.Clock
  direction: GZ_TO_ROS
  lazy: false
```

Set the bridge `override_frame_id` parameter to `camera_color_optical_frame`.

- [ ] **Step 4: Implement launch ordering**

Declare `gui`, `world`, `camera_xyz`, `camera_rpy`, and `verbose`. Generate robot description; start robot-state publisher on `/piper_pika/joint_states`; run Gazebo with `-r` and add `-s` for headless mode; spawn from `/robot_description`; start bridge, adapter, and `/joint_states` relay; then spawn `joint_state_broadcaster`, `arm_controller`, and `pika_gripper_controller` with a 60-second controller-manager timeout. Never create `ros2_control_node`.

- [ ] **Step 5: Add the hardware convenience script**

```bash
#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" docker compose run --rm piper_ros2 bash -lc '
    source /workspace/ros2_ws/install/setup.bash
    exec ros2 launch piper_elevator_gazebo gazebo_hardware.launch.py "$@"
' bash "$@"
```

- [ ] **Step 6: Build and verify**

```bash
./scripts/build.sh
docker compose run --rm piper_ros2 bash -lc '
  source /workspace/ros2_ws/install/setup.bash
  ros2 launch piper_elevator_gazebo gazebo_hardware.launch.py --show-args
  cd /workspace/ros2_ws/src/piper_elevator_gazebo
  pytest -q test/test_launch_boundary.py
'
```

Expected: arguments print and the isolation test passes.

- [ ] **Step 7: Commit**

```bash
git add scripts/gazebo_hardware.sh ros2_ws/src/piper_elevator_gazebo
git commit -m "feat: launch Gazebo virtual hardware only"
```

---

### Task 7: Connect separately launched core nodes to external hardware

**Files:**
- Create: `ros2_ws/src/piper_elevator_app/piper_elevator_app/launch_mode.py`
- Create: `ros2_ws/src/piper_elevator_app/test/test_launch_mode.py`
- Modify: `ros2_ws/src/piper_elevator_app/launch/piper_pika_moveit.launch.py`
- Modify: `ros2_ws/src/piper_elevator_app/launch/button_detector.launch.py`
- Modify: `ros2_ws/src/piper_elevator_app/launch/button_approach_planner.launch.py`

**Interfaces:**
- Consumes: normalized joints, arm trajectory action, clock, and description-only xacro.
- Produces: core launches that use simulated time without Gazebo-specific algorithm branches.

- [ ] **Step 1: Write failing launch-policy tests**

```python
from piper_elevator_app.launch_mode import select_moveit_launch_mode


def test_external_hardware_uses_description_only():
    mode = select_moveit_launch_mode(True)
    assert mode.description_file == 'piper_pika_description.urdf.xacro'
    assert mode.start_robot_state_publisher is False
    assert mode.start_ros2_control is False
    assert mode.start_controller_spawners is False
    assert mode.default_joint_states_topic == '/piper_pika/joint_states'


def test_legacy_mode_preserves_existing_proxy():
    mode = select_moveit_launch_mode(False)
    assert mode.description_file == 'piper_pika.urdf.xacro'
    assert mode.start_robot_state_publisher is True
    assert mode.start_ros2_control is True
    assert mode.start_controller_spawners is True
    assert mode.default_joint_states_topic == 'control/joint_states'
```

- [ ] **Step 2: Verify import failure**

Run the focused test. Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement immutable policy**

Create frozen dataclass `MoveItLaunchMode` with the five fields above and `select_moveit_launch_mode(external_hardware: bool)` returning the two tested configurations.

- [ ] **Step 4: Apply external mode in MoveIt launch**

Add `external_hardware:=false` and `use_sim_time:=false`. Select the correct description file in `OpaqueFunction`. External mode omits RSP, `ros2_control_node`, and spawners; legacy mode preserves them. Always start move_group, conditionally RViz, and pass typed `use_sim_time` to both.

- [ ] **Step 5: Add simulated time to detector and planner launches**

Declare `use_sim_time` default false and pass:

```python
ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
```

Do not add a Gazebo parameter to application nodes.

- [ ] **Step 6: Run unit, interface, and legacy checks**

```bash
docker compose run --rm piper_ros2 bash -lc '
  source /workspace/ros2_ws/install/setup.bash
  cd /workspace/ros2_ws/src/piper_elevator_app
  pytest -q test/test_launch_mode.py test/test_robot_description.py
  ros2 launch piper_elevator_app piper_pika_moveit.launch.py --show-args | grep external_hardware
  ros2 launch piper_elevator_app button_detector.launch.py --show-args | grep use_sim_time
  ros2 launch piper_elevator_app button_approach_planner.launch.py --show-args | grep use_sim_time
'
./scripts/build.sh
./scripts/check.sh
```

Expected: all pass and legacy MoveIt remains ready.

- [ ] **Step 7: Commit**

```bash
git add ros2_ws/src/piper_elevator_app
git commit -m "feat: connect MoveIt core to external hardware"
```

---

### Task 8: Add a headless virtual-hardware integration check

**Files:**
- Create: `ros2_ws/src/piper_elevator_gazebo/test/virtual_hardware_probe.py`
- Create: `scripts/check_gazebo.sh`

**Interfaces:**
- Consumes: hardware-only launch and normalized interfaces.
- Produces: repeatable evidence for controller activation, joint motion, camera contract, clock, and core-node absence.

- [ ] **Step 1: Write the running-system probe**

Create an `rclpy` program subscribing with sensor QoS to joint states, color, aligned depth, and CameraInfo. Wait 45 seconds. Require 848 by 480 images, `16UC1` depth, and `camera_color_optical_frame`. Require the latest color, depth, and CameraInfo timestamps to differ by no more than 0.08 seconds, matching the detector's current synchronization slop. Send `[0.0, 0.10, -0.10, 0.0, 0.0, 0.0]` over 3 seconds to `/arm_controller/follow_joint_trajectory`; require success and final joint2/joint3 errors below 0.03 radian.

- [ ] **Step 2: Verify timeout without hardware**

Run the probe alone. Expected: nonzero exit naming missing streams.

- [ ] **Step 3: Create `check_gazebo.sh`**

Start `gazebo_hardware.launch.py gui:=false verbose:=2` in the background with a cleanup trap. Wait for controller services; require all three controllers active. Reject move_group, detector, planner, and mock-pose nodes. Expand the installed Gazebo xacro, require GazeboSimSystem, reject GenericSystem, then run the probe.

- [ ] **Step 4: Run the check**

```bash
./scripts/check_gazebo.sh
```

Expected: the script reports plugin isolation, three active controllers, valid RGB-D contract, successful physical joint trajectory, and no core nodes.

- [ ] **Step 5: Commit**

```bash
git add scripts/check_gazebo.sh ros2_ws/src/piper_elevator_gazebo/test/virtual_hardware_probe.py
git commit -m "test: verify Gazebo virtual hardware contract"
```

---

### Task 9: Pass the existing YOLO path with the rendered `up` button

**Files:**
- Create: `ros2_ws/src/piper_elevator_gazebo/test/perception_probe.py`
- Create: `scripts/check_gazebo_perception.sh`
- Modify based on rendered evidence: panel model, up mesh, world lighting, or camera xacro.

**Interfaces:**
- Consumes: unchanged detector, rendered RGB-D, TF, and test-only coordinate `(0.55, 0.00, 0.30)`.
- Produces: stable YOLO `/button_pose` for class `up`, without a ground-truth publisher.

- [ ] **Step 1: Write the acceptance probe**

Publish `String(data='up')` to `/button_selection`. Wait 60 seconds for selected `up`, valid detection, optical-frame pose, and depth 0.1–2.0 m. Transform the detected pose into `world` via `tf2_ros.Buffer` and require Euclidean error below 0.05 m from `(0.55, 0.00, 0.30)`. Never publish that constant.

- [ ] **Step 2: Create the perception check**

Start hardware headlessly and separately start:

```bash
ros2 launch piper_elevator_app button_detector.launch.py use_sim_time:=true
```

Run the probe, clean up both processes, save one color and debug frame under `/tmp`, and reject `/mock_button_pose`.

- [ ] **Step 3: Run the first acceptance attempt**

```bash
./scripts/check_gazebo_perception.sh
```

Expected: either pass or a concrete no-detection/pose-error result with two saved images.

- [ ] **Step 4: Tune only bounded virtual visual values**

Keep detector source, model, and 0.60 confidence threshold unchanged. Based on each saved frame, change one value per run within: panel x 0.50–0.65 m, panel z 0.25–0.40 m, light intensity 0.6–1.4, metal roughness 0.20–0.45, button diameter 0.030–0.050 m, camera translation within 30 mm of default, and arrow RGB 0.01–0.15. Rerun the same probe after each change. Do not add a pose publisher.

- [ ] **Step 5: Require three clean passes**

Run `./scripts/check_gazebo_perception.sh` three times. Expected: all pass with error below 0.05 m.

- [ ] **Step 6: Commit**

```bash
git add scripts/check_gazebo_perception.sh ros2_ws/src/piper_elevator_gazebo
git commit -m "test: validate YOLO against Gazebo button scene"
```

---

### Task 10: Verify separately launched planning and approach execution

**Files:**
- Create: `scripts/check_gazebo_approach.sh`

**Interfaces:**
- Consumes: virtual hardware, external MoveIt, YOLO pose, and planner services.
- Produces: full-flow evidence without changing the hardware-only launcher.

- [ ] **Step 1: Create the acceptance script**

Start these independent launches under one cleanup trap:

```bash
ros2 launch piper_elevator_gazebo gazebo_hardware.launch.py gui:=false
ros2 launch piper_elevator_app piper_pika_moveit.launch.py \
  external_hardware:=true use_sim_time:=true use_rviz:=false
ros2 launch piper_elevator_app button_detector.launch.py use_sim_time:=true
ros2 launch piper_elevator_app button_approach_planner.launch.py \
  use_sim_time:=true simulation_mode:=true \
  camera_calibration_valid:=true allow_execution:=true \
  publish_camera_tf:=false
```

Wait for controllers, `/move_action`, planner services, and valid selected `up`. Call Plan then Execute; require `success: true` and final status containing `succeeded`. Reject mock-pose nodes.

- [ ] **Step 2: Run the full flow**

```bash
./scripts/check_gazebo_approach.sh
```

Expected: YOLO supplies the target, MoveIt plans, and Gazebo reaches the pre-contact pose; the button does not move.

- [ ] **Step 3: Recheck hardware isolation**

Run `./scripts/check_gazebo.sh`. Expected: still no core nodes in hardware-only bringup.

- [ ] **Step 4: Commit**

```bash
git add scripts/check_gazebo_approach.sh
git commit -m "test: verify Gazebo button approach flow"
```

---

### Task 11: Replace FakeSystem simulation guidance and document migration

**Files:**
- Modify: `scripts/sim.sh`
- Modify: `develop.md`
- Modify: `README.md`
- Modify: `scripts/check.sh`

**Interfaces:**
- Consumes: completed launch and check commands.
- Produces: a supported Gazebo operator workflow while preserving the real launch.

- [ ] **Step 1: Redirect the generic simulation command**

Replace the FakeSystem demo in `scripts/sim.sh` with:

```bash
exec "${project_root}/scripts/gazebo_hardware.sh" "$@"
```

- [ ] **Step 2: Update concise development commands**

Preserve existing topic tables in `develop.md`, but use four terminals:

```bash
./scripts/gazebo_hardware.sh
ros2 launch piper_elevator_app piper_pika_moveit.launch.py \
  external_hardware:=true use_sim_time:=true
ros2 launch piper_elevator_app button_detector.launch.py use_sim_time:=true
ros2 launch piper_elevator_app button_approach_planner.launch.py \
  use_sim_time:=true simulation_mode:=true \
  camera_calibration_valid:=true allow_execution:=true
```

Use ROS domain 0 unless every terminal explicitly shares another value. Keep Plan, Execute, and Clear commands.

- [ ] **Step 3: Update README**

Document Fortress dependencies, hardware-only boundary, camera/joint/action parity, separate core launches, unchanged real proxy, hand-eye and CAN safety requirements, and the non-goal of button activation. Mark `button_approach_sim.sh` and its launch as legacy diagnostics rather than the supported workflow; do not delete them.

- [ ] **Step 4: Extend static checks**

Add executable and launch checks:

```bash
ros2 pkg executables piper_elevator_gazebo | grep -F \
  "piper_elevator_gazebo depth_adapter" >/dev/null
ros2 launch piper_elevator_gazebo gazebo_hardware.launch.py \
  --show-args >/dev/null
```

Keep the legacy MoveIt smoke check for real-path regression protection.

- [ ] **Step 5: Run the complete suite**

```bash
./scripts/build.sh
./scripts/check.sh
./scripts/check_gazebo.sh
./scripts/check_gazebo_perception.sh
./scripts/check_gazebo_approach.sh
```

Expected: all exit 0. Confirm `AI_PROJECT_CONTEXT.md` remains unstaged.

- [ ] **Step 6: Commit**

```bash
git add scripts/sim.sh scripts/check.sh develop.md README.md
git commit -m "docs: switch simulation workflow to Gazebo"
```

---

### Task 12: Final regression and evidence handoff

**Files:**
- Verify only; repair the responsible task file if a check exposes a defect.

**Interfaces:**
- Consumes: every artifact and check from Tasks 1–11.
- Produces: reproducible evidence for the approved design.

- [ ] **Step 1: Run package tests with testing enabled**

```bash
docker compose run --rm piper_ros2 bash -lc '
  set -e
  cd /workspace/ros2_ws
  source /opt/ros/humble/setup.bash
  colcon build --symlink-install --cmake-args -DBUILD_TESTING=ON
  source install/setup.bash
  colcon test --packages-select piper_elevator_app piper_elevator_gazebo \
    --event-handlers console_direct+
  colcon test-result --verbose
'
```

Expected: zero failed tests.

- [ ] **Step 2: Run Python style checks**

```bash
docker compose run --rm piper_ros2 bash -lc '
  source /workspace/ros2_ws/install/setup.bash
  ament_flake8 \
    /workspace/ros2_ws/src/piper_elevator_app/piper_elevator_app \
    /workspace/ros2_ws/src/piper_elevator_app/test \
    /workspace/ros2_ws/src/piper_elevator_gazebo/piper_elevator_gazebo \
    /workspace/ros2_ws/src/piper_elevator_gazebo/test
  ament_pep257 \
    /workspace/ros2_ws/src/piper_elevator_app/piper_elevator_app \
    /workspace/ros2_ws/src/piper_elevator_gazebo/piper_elevator_gazebo
'
```

Expected: exit code 0.

- [ ] **Step 3: Repeat critical runtime checks**

```bash
./scripts/check_gazebo.sh
./scripts/check_gazebo_approach.sh
```

Expected: physical controller motion and YOLO-to-approach both pass.

- [ ] **Step 4: Inspect the final change set**

```bash
git status --short
git diff --check HEAD~11..HEAD
git log --oneline -12
```

Expected: no whitespace errors; only the user's pre-existing `AI_PROJECT_CONTEXT.md` deletion may remain outside task commits.

- [ ] **Step 5: Report handoff evidence**

Report the virtual-hardware command; three camera topics; control action; five verification results; measured joint error; YOLO confidence and 3-D error; absence of GenericSystem from Gazebo xacro; real-launch regression result; and that button activation is not simulated.
