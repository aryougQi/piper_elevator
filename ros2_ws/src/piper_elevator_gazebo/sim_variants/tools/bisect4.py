import time, threading
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, CameraInfo, JointState
from trajectory_msgs.msg import JointTrajectoryPoint

JOINTS = tuple(f'joint{i}' for i in range(1, 7))

rclpy.init()
node = Node('bisect4')
joints = {}
def on_js(m):
    v = dict(zip(m.name, m.position))
    if all(j in v for j in JOINTS):
        joints.update({j: float(v[j]) for j in JOINTS})
node.create_subscription(JointState, '/piper_pika/joint_states', on_js, 10)
node.create_subscription(Image, '/camera/color/image_raw',
                         lambda m: None, qos_profile_sensor_data)
node.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw',
                         lambda m: None, qos_profile_sensor_data)
node.create_subscription(CameraInfo, '/camera/color/camera_info',
                         lambda m: None, qos_profile_sensor_data)
node.create_subscription(JointState, '/elevator_button/joint_states',
                         lambda m: None, 10)
arm = ActionClient(node, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')

def wait_future(fut, timeout):
    t0 = time.monotonic()
    while not fut.done() and time.monotonic() - t0 < timeout:
        time.sleep(0.05)
    if not fut.done():
        raise TimeoutError('future timeout')
    return fut.result()

executor = MultiThreadedExecutor()
executor.add_node(node)
threading.Thread(target=executor.spin, daemon=True).start()

deadline = time.monotonic() + 30
while time.monotonic() < deadline and (not joints or not arm.wait_for_server(timeout_sec=1.0)):
    time.sleep(0.1)
assert joints, 'no joint states'

for delta in (0.15, -0.15):
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(JOINTS)
    p = JointTrajectoryPoint()
    p.positions = [joints[j] + delta if j == 'joint1' else joints[j] for j in JOINTS]
    p.time_from_start = Duration(seconds=2.5).to_msg()
    goal.trajectory.points = [p]
    t0 = time.monotonic()
    handle = wait_future(arm.send_goal_async(goal), 15)
    assert handle.accepted
    response = wait_future(handle.get_result_async(), 30)
    print(f'move {delta:+.2f}: elapsed={round(time.monotonic()-t0,1)} '
          f'error_code={response.result.error_code}')
executor.shutdown()
node.destroy_node(); rclpy.shutdown()
print('BOTH_MOVES_OK')
