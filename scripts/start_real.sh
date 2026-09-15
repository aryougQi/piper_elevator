#!/usr/bin/env bash
# Complete real-hardware stack. No task command is sent by this script.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

usage() {
    cat <<'HELP'
用法：./scripts/start_real.sh [选项] [name:=value ...]

默认：CAN 1 Mbps、ROS_DOMAIN_ID=0、D405 315122272433、整套实机节点；不使能运动。
  --execute                 使能机械臂并开放运动；不会自动发送按压任务
  --no-rviz                 不启动 RViz
  --camera-serial SERIAL    相机序列号（可省略前导下划线）
  --can-interface NAME      CAN 接口，默认 can0
  --bitrate RATE            CAN 速率，默认 1000000
  --dry-run                 仅打印配置和启动命令，不检查/操作硬件
  -h, --help                显示帮助

可透传 camera_x:=...、start_camera:=false 等 ROS launch 参数。
auto_enable/hardware_commands_enabled/allow_execution 由 --execute 统一控制。
默认加载 config/real_handeye.json 中用户选定的历史外参并发布相机 TF。
--execute 自动开放该外参的执行准入；命令行外参仍可覆盖配置。
力矩阈值仍由 button_press.yaml 配置，本脚本不修改任何标定状态。
HELP
}

fail() { echo "错误：$*" >&2; exit 1; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || fail "$1 缺少参数"; }

execute=false
dry_run=false
can_interface=can0
bitrate=1000000
camera_serial=_315122272433
rviz=true
extra=()
declare -A supplied=()
while (($#)); do
    case "$1" in
        --execute) execute=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        --no-rviz) rviz=false; shift ;;
        --camera-serial) need_value "$@"; camera_serial="_${2#_}"; shift 2 ;;
        --can-interface) need_value "$@"; can_interface="$2"; shift 2 ;;
        --bitrate) need_value "$@"; bitrate="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *:=*)
            name="${1%%:=*}"
            value="${1#*:=}"
            [[ "$name" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ && -n "$value" ]] || fail "无效参数：$1"
            case "$name" in
                simulation_mode|use_sim_time|auto_enable|hardware_commands_enabled|allow_execution)
                    fail "$name 由实机脚本统一管理；开放运动请使用 --execute" ;;
                can_port) can_interface="$value" ;;
                camera_serial_no) camera_serial="_${value#_}" ;;
                use_rviz)
                    [[ "$value" == true || "$value" == false ]] || fail 'use_rviz 必须为 true/false'
                    rviz="$value" ;;
                *) extra+=("$1"); supplied["$name"]="$value" ;;
            esac
            shift ;;
        *) fail "未知选项：$1（使用 --help 查看用法）" ;;
    esac
done

[[ "$can_interface" =~ ^[a-zA-Z0-9_.-]{1,15}$ ]] || fail '无效 CAN 接口名'
[[ "$bitrate" =~ ^[1-9][0-9]{0,6}$ ]] || fail '无效 CAN 速率'
[[ "$camera_serial" =~ ^_[0-9]+$ ]] || fail '相机序列号应为数字'
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
[[ "$ROS_DOMAIN_ID" =~ ^[0-9]+$ ]] || fail 'ROS_DOMAIN_ID 应为非负整数'
# User-selected historical profile; load numbers as data, never shell code.
profile_lines="$(python3 - "$project_root/config/real_handeye.json" <<'PYPROFILE'
import json, math, sys
with open(sys.argv[1]) as stream:
    profile = json.load(stream)
assert profile['parent_frame'] == 'tcp_link' and profile['child_frame'] == 'camera_link'
assert len(profile['xyz']) == len(profile['rpy']) == 3
values = profile['xyz'] + profile['rpy']
assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
print(profile['camera_serial'])
for name, value in zip(('camera_x', 'camera_y', 'camera_z', 'camera_roll', 'camera_pitch', 'camera_yaw'), values):
    print(f'{name}:={value}')
PYPROFILE
)" || fail '无法读取 config/real_handeye.json'
mapfile -t profile_values <<< "$profile_lines"
if "$execute"; then
    [[ "${supplied[camera_calibration_valid]:-true}" == true ]] || fail '执行模式不能显式关闭 camera_calibration_valid'
    if [[ "${camera_serial#_}" != "${profile_values[0]}" ]]; then
        for name in camera_x camera_y camera_z camera_roll camera_pitch camera_yaw; do
            [[ -n "${supplied[$name]:-}" ]] || fail "相机与保存外参不匹配；请提供对应相机的六个外参（缺少 $name）"
        done
    fi
fi

launch_args=(
    simulation_mode:=false use_sim_time:=false
    "can_port:=$can_interface" "camera_serial_no:=$camera_serial"
    start_camera:=true start_pika_driver:=false "use_rviz:=$rviz"
    speed_percent:=10 publish_camera_tf:=true "camera_calibration_valid:=$execute"
    "${profile_values[@]:1}"
    "${extra[@]}"
    "auto_enable:=$execute" "hardware_commands_enabled:=$execute" "allow_execution:=$execute"
)
command=(docker compose run --rm piper_ros2 bash -lc '
    set -e
    source /workspace/ros2_ws/install/setup.bash
    exec ros2 launch piper_elevator_app elevator_task.launch.py "$@"
' bash "${launch_args[@]}")

echo "实机：CAN=$can_interface ($bitrate bps)，ROS_DOMAIN_ID=$ROS_DOMAIN_ID，相机=$camera_serial，运动=$execute"
echo "外参：config/real_handeye.json（用户选定的 2026-09-02 历史结果）"
if "$dry_run"; then
    echo '将检查 CAN；仅在接口为 DOWN 时配置速率并启用，已 UP 的接口不重置。'
    printf 'ROS_DOMAIN_ID=%q ' "$ROS_DOMAIN_ID"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

for executable in ip python3 docker flock; do
    command -v "$executable" >/dev/null || fail "缺少命令 $executable"
done
[[ -r ros2_ws/install/setup.bash ]] || fail '未编译工作空间，请先运行 ./scripts/build.sh'
docker compose version >/dev/null
docker info >/dev/null || fail 'Docker 不可用，请检查服务及当前用户权限'
if "$rviz"; then
    [[ -n "${DISPLAY:-}" ]] || fail '没有 DISPLAY；无界面运行请加 --no-rviz'
fi

# This lock covers instances of this wrapper. The existing RealSense launch
# separately rejects competing camera owners via its host-IPC lock.
lock_dir="${project_root}/ros2_ws/diagnostics/data/start_real"
mkdir -p "$lock_dir"
exec 9>"$lock_dir/$can_interface.lock"
flock -n 9 || fail "已有 start_real.sh 使用 $can_interface，请先退出旧实例"

can_json="$(ip -json -details link show dev "$can_interface")" || fail "找不到接口 $can_interface"
can_state="$(CAN_LINK_JSON="$can_json" python3 - "$bitrate" <<'PY'
import json, os, sys
link = json.loads(os.environ['CAN_LINK_JSON'])[0]
info = link.get('linkinfo', {})
if info.get('info_kind') != 'can':
    sys.exit('指定接口不是 CAN')
data = info.get('info_data', {})
if 'UP' in link.get('flags', []):
    if data.get('bittiming', {}).get('bitrate') != int(sys.argv[1]):
        sys.exit('CAN 已 UP 但速率不匹配；请先停止占用程序并手动配置')
    if data.get('state') in ('BUS-OFF', 'STOPPED', 'SLEEPING'):
        sys.exit('CAN 状态异常：' + str(data.get('state')))
    print('ready')
else:
    print('down')
PY
)" || fail 'CAN 检查失败'

if [[ "$can_state" == down ]]; then
    privilege=()
    if ((EUID != 0)); then
        command -v sudo >/dev/null || fail '初始化 CAN 需要 sudo'
        privilege=(sudo)
    fi
    "${privilege[@]}" ip link set dev "$can_interface" type can bitrate "$bitrate"
    "${privilege[@]}" ip link set dev "$can_interface" up
fi
ip -details -statistics link show dev "$can_interface"
echo '启动整套实机节点；Ctrl+C 退出。未发送任何按压任务。'
echo '新视觉实验模块不会由此脚本启动。'
exec "${command[@]}"
