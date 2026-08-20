#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

docker compose run --rm piper_ros2 bash -lc '
    set -eo pipefail
    cd /workspace/ros2_ws
    source install/setup.bash
    set -u

    expected_packages=(
        agx_arm_ctrl
        agx_arm_description
        agx_arm_moveit
        agx_arm_msgs
        data_msgs
        data_tools
        piper_elevator_app
        pika_remote_agx_arm
        realsense2_camera
        realsense2_camera_msgs
        realsense2_description
        sensor_tools
    )

    for package in "${expected_packages[@]}"; do
        ros2 pkg prefix "${package}" >/dev/null
    done

    ros2 pkg executables sensor_tools | grep -F "sensor_tools serial_gripper_imu" >/dev/null
    ros2 pkg executables sensor_tools | grep -F "sensor_tools usb_camera.py" >/dev/null
    ros2 pkg executables piper_elevator_app | grep -F "piper_elevator_app button_detector" >/dev/null
    model_path="$(ros2 pkg prefix piper_elevator_app)/share/piper_elevator_app/models/elevator_buttons_yolov10s.onnx"
    test -s "${model_path}"
    MODEL_PATH="${model_path}" python3 -c "import os, numpy as np, onnxruntime as ort; assert \"CUDAExecutionProvider\" in ort.get_available_providers(), ort.get_available_providers(); session = ort.InferenceSession(os.environ[\"MODEL_PATH\"], providers=[\"CUDAExecutionProvider\", \"CPUExecutionProvider\"]); assert session.get_providers()[0] == \"CUDAExecutionProvider\", session.get_providers(); input_meta = session.get_inputs()[0]; session.run(None, {input_meta.name: np.zeros(input_meta.shape, dtype=np.float32)})"
    ros2 launch piper_elevator_app button_detector.launch.py --show-args >/dev/null
    ros2 launch sensor_tools open_single_gripper.launch.py --show-args >/dev/null
    ros2 launch realsense2_camera rs_launch.py --show-args >/dev/null

    launch_log=/tmp/piper_moveit_smoke.log
    ros2 launch agx_arm_moveit demo.launch.py \
        arm_type:=piper \
        effector_type:=none \
        use_rviz:=false \
        >"${launch_log}" 2>&1 &
    launch_pid=$!

    cleanup() {
        if kill -0 "${launch_pid}" 2>/dev/null; then
            kill -TERM "${launch_pid}" 2>/dev/null || true
            for _ in $(seq 1 10); do
                if ! kill -0 "${launch_pid}" 2>/dev/null; then
                    break
                fi
                sleep 0.5
            done
            kill -KILL "${launch_pid}" 2>/dev/null || true
        fi
        wait "${launch_pid}" 2>/dev/null || true
    }
    trap cleanup EXIT

    ready=false
    for _ in $(seq 1 30); do
        if grep -Fq "You can start planning now!" "${launch_log}" \
            && grep -Eq "\[spawner_joint_state_broadcaster\].*Configured and activated" "${launch_log}" \
            && grep -Eq "\[spawner_arm_controller\].*Configured and activated" "${launch_log}"; then
            ready=true
            break
        fi
        if ! kill -0 "${launch_pid}" 2>/dev/null; then
            break
        fi
        sleep 1
    done

    if [[ "${ready}" != true ]]; then
        cat "${launch_log}" >&2
        echo "Piper MoveIt did not become ready." >&2
        exit 1
    fi

    echo "OK: 12 ROS 2 packages are installed."
    echo "OK: Pika gripper and RealSense launch files are loadable."
    echo "OK: YOLO ONNX model runs with CUDAExecutionProvider."
    echo "OK: Piper MoveIt, ros2_control and mock controllers are running."
'
