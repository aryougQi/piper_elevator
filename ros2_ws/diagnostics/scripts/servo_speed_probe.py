import json,time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TwistStamped
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState
rclpy.init(); n=Node('servo_speed_probe'); values={}; last={}
def emit(key,data):
 now=time.monotonic()
 if now-last.get(key,0)>.5: print(json.dumps({'time':time.time(),'kind':key,**data}),flush=True);last[key]=now
def state(m):
 values.update(zip(m.name,m.position))
 emit('state',{'position':dict(zip(m.name,m.position))})
def traj(m,key):
 if not m.points:return
 p=m.points[-1]
 emit(key,{'lead':[v-values.get(k,v) for k,v in zip(m.joint_names,p.positions)],'duration':p.time_from_start.sec+p.time_from_start.nanosec/1e9})
n.create_subscription(JointState,'/piper_pika/joint_states',state,qos_profile_sensor_data)
n.create_subscription(TwistStamped,'/servo_node/delta_twist_cmds',lambda m:emit('twist',{'linear':[m.twist.linear.x,m.twist.linear.y,m.twist.linear.z],'angular':[m.twist.angular.x,m.twist.angular.y,m.twist.angular.z]}),10)
for key,topic in [('raw','/servo_node/raw_joint_trajectory'),('output','/arm_controller/joint_trajectory')]:n.create_subscription(JointTrajectory,topic,lambda m,k=key:traj(m,k),10)
try:rclpy.spin(n)
finally:n.destroy_node();rclpy.shutdown()
