#!/usr/bin/env bash
# Manage the headless Gazebo session inside the sim_collect container.
# Usage: sim_session.sh start <LIGHT_ID> | stop | status
#
# pkill patterns use the [x] character-class trick so the wrapper shell's
# own command line never matches itself.

set -euo pipefail

CONTAINER=sim_collect
DOMAIN=77
VARIANTS_DIR=/workspace/ros2_ws/src/piper_elevator_gazebo/sim_variants

exec_in() {
    docker exec -e ROS_DOMAIN_ID=$DOMAIN "$CONTAINER" bash -lc "$1"
}

do_stop() {
    docker exec -e ROS_DOMAIN_ID=$DOMAIN "$CONTAINER" bash -c "
        pkill -f '[r]os2 launch.*gazebo' 2>/dev/null || true
        pkill -f '[i]gn gazebo' 2>/dev/null || true
        pkill -f '[r]uby /usr/bin/ign' 2>/dev/null || true
        sleep 3
        pkill -9 -f '[r]os2 launch.*gazebo' 2>/dev/null || true
        pkill -9 -f '[i]gn gazebo' 2>/dev/null || true
        pkill -9 -f '[r]uby /usr/bin/ign' 2>/dev/null || true
        sleep 1
        echo stop_done"
    sleep 1
}

do_start() {
    local light=$1
    do_stop
    docker exec -e ROS_DOMAIN_ID=$DOMAIN \
        -e GZ_SIM_RESOURCE_PATH="$VARIANTS_DIR/models" \
        -e IGN_GAZEBO_RESOURCE_PATH="$VARIANTS_DIR/models" \
        "$CONTAINER" bash -c "
        source /workspace/ros2_ws/install/setup.bash
        nohup ros2 launch $VARIANTS_DIR/tools/collect_launch.py \
            gui:=false world:=$VARIANTS_DIR/worlds/light_${light}.sdf \
            > /tmp/gazebo_${light}.log 2>&1 &
        echo launched_pid_\$!"
    # Single in-container session waits for the world to finish loading.
    docker exec -e ROS_DOMAIN_ID=$DOMAIN "$CONTAINER" bash -c "
        source /workspace/ros2_ws/install/setup.bash
        for i in \$(seq 1 90); do
            sleep 2
            if timeout 3 ros2 topic list 2>/dev/null | grep -q '/elevator_button/joint_states'; then
                sleep 6
                echo READY_after_\${i}polls
                exit 0
            fi
        done
        echo NOT_READY
        tail -20 /tmp/gazebo_${light}.log
        exit 1"
}

case "${1:-}" in
    start) do_start "${2:?light id required}" ;;
    stop) do_stop; echo "gazebo stopped" ;;
    status)
        exec_in "pgrep -af '[i]gn gazebo' | head -3 || echo 'not running'"
        ;;
    *) echo "usage: $0 start <LIGHT_ID> | stop | status" >&2; exit 1 ;;
esac
