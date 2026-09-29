"""Bisect which collector component breaks the action result."""
import sys, time
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, CameraInfo, JointState
from trajectory_msgs.msg import JointTrajectoryPoint

JOINTS = tuple(f'joint{i}' for i in range(1, 7))
flags = set(sys.argv[1:])
USE_SIM = 'sim' in flags
WITH_IMGS = 'imgs' in flags
WITH_TF = 'tf' in flags

rclpy.init()
node = Node('bisect2')
if USE_SIM:
    node.set_parameters([Parameter('use_sim_time', value=True)])
joints = {}
def on_js(m):
    v = dict(zip(m.name, m.position))
    if all(j in v for j in JOINTS):
        joints.update({j: float(v[j]) for j in JOINTS})
node.create_subscription(JointState, '/piper_pika/joint_states', on_js, 10)

if WITH_IMGS:
    node.create_subscription(Image, '/camera/color/image_raw',
                             lambda m: None, qos_profile_sensor_data)
    node.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw',
                             lambda m: None, qos_profile_sensor_data)
    node.create_subscription(CameraInfo, '/camera/color/camera_info',
                             lambda m: None, qos_profile_sensor_data)
    node.create_subscription(JointState, '/elevator_button/joint_states',
                             lambda m: None, 10)

tf_helper = None
if WITH_TF:
    from tf2_ros import Buffer, TransformListener
    tf_helper = Node('bisect_tf')
    buf = Buffer()
    TransformListener(buf, tf_helper, spin_thread=True)

arm = ActionClient(node, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
deadline = time.monotonic() + 30
while time.monotonic() < deadline and (not joints or not arm.wait_for_server(timeout_sec=1.0)):
    rclpy.spin_once(node, timeout_sec=0.1)

goal = FollowJointTrajectory.Goal()
goal.trajectory.joint_names = list(JOINTS)
p = JointTrajectoryPoint()
p.positions = [joints[j] - 0.15 if j == 'joint1' else joints[j] for j in JOINTS]
p.time_from_start = Duration(seconds=2.5).to_msg()
goal.trajectory.points = [p]
sent = arm.send_goal_async(goal)
while not sent.done():
    rclpy.spin_once(node, timeout_sec=0.05)
handle = sent.result()
result = handle.get_result_async()
t0 = time.monotonic()
while not result.done() and time.monotonic() - t0 < 15:
    rclpy.spin_once(node, timeout_sec=0.05)
print(f'flags={sorted(flags)} accepted={handle.accepted} '
      f'result_done={result.done()} elapsed={round(time.monotonic()-t0,1)}')
node.destroy_node()
rclpy.shutdown()
