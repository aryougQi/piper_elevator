import sys, time
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState

JOINTS = tuple(f'joint{i}' for i in range(1, 7))
USE_SIM_TIME = sys.argv[1] == '1'

rclpy.init()
node = Node('bisect_move')
if USE_SIM_TIME:
    node.set_parameters([Parameter('use_sim_time', value=True)])
joints = {}
def on_js(m):
    v = dict(zip(m.name, m.position))
    if all(j in v for j in JOINTS):
        joints.update({j: float(v[j]) for j in JOINTS})
node.create_subscription(JointState, '/piper_pika/joint_states', on_js, 10)
arm = ActionClient(node, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')

deadline = time.monotonic() + 30
while time.monotonic() < deadline and (not joints or not arm.wait_for_server(timeout_sec=1.0)):
    rclpy.spin_once(node, timeout_sec=0.1)
print('ready, joints:', {k: round(v,3) for k,v in joints.items()})

goal = FollowJointTrajectory.Goal()
goal.trajectory.joint_names = list(JOINTS)
p = JointTrajectoryPoint()
p.positions = [joints[j] + 0.15 if j == 'joint1' else joints[j] for j in JOINTS]
p.time_from_start = Duration(seconds=2.5).to_msg()
goal.trajectory.points = [p]
sent = arm.send_goal_async(goal)
while not sent.done():
    rclpy.spin_once(node, timeout_sec=0.05)
handle = sent.result()
print('accepted:', handle.accepted)
result = handle.get_result_async()
t0 = time.monotonic()
while not result.done() and time.monotonic() - t0 < 20:
    rclpy.spin_once(node, timeout_sec=0.05)
print('use_sim_time=', USE_SIM_TIME, 'result done:', result.done(),
      'elapsed:', round(time.monotonic()-t0, 1))
if result.done():
    print('error_code:', result.result().result.error_code)
node.destroy_node(); rclpy.shutdown()
