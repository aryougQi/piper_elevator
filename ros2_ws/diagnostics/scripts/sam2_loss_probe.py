#!/usr/bin/env python3
"""Disable tracker output during active Gazebo Servo and verify fail-closed recovery."""
import json
import os
from pathlib import Path
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped

if os.environ.get('ROS_DOMAIN_ID') != '73':
    raise SystemExit('This diagnostic is restricted to the isolated simulation domain 73')
rclpy.init();n=Node('sam2_loss_probe');state={'injected':False,'press_seen':False,'result':None}
client=n.create_client(SetParameters,'/sam2_button_tracker/set_parameters')
publisher=n.create_publisher(String,'/elevator_task/command',10)

def set_enabled(value):
    req=SetParameters.Request(parameters=[Parameter(name='enabled',value=ParameterValue(type=ParameterType.PARAMETER_BOOL,bool_value=value))])
    return client.call_async(req)

def visual(msg):
    if not state['injected'] and msg.data.startswith('VISUAL_FINAL_APPROACH'):
        state['injected']=True;state['injected_at']=time.monotonic()
        state['disable_future']=set_enabled(False)

def phase(msg):
    if state['injected'] and msg.data.startswith('PRESSING'):state['press_seen']=True

def result(msg):
    if state['injected']:state['result']=msg.data

def twist(msg):
    if not state['injected']:return
    t=msg.twist
    moving=max(abs(v) for v in [t.linear.x,t.linear.y,t.linear.z,t.angular.x,t.angular.y,t.angular.z])>1e-6
    if moving:state['last_motion_after_injection']=time.monotonic()-state['injected_at']

q=QoSProfile(depth=10,durability=DurabilityPolicy.TRANSIENT_LOCAL)
n.create_subscription(String,'/button_visual_servo/status',visual,q)
n.create_subscription(String,'/elevator_task/status',phase,q)
n.create_subscription(String,'/elevator_task/result',result,q)
n.create_subscription(TwistStamped,'/servo_node/delta_twist_cmds',twist,10)
try:
    if not client.wait_for_service(timeout_sec=15):raise RuntimeError('tracker unavailable')
    start=time.monotonic()
    while time.monotonic()-start<1:rclpy.spin_once(n,timeout_sec=.1)
    publisher.publish(String(data='press up'))
    while time.monotonic()-start<180 and state['result'] is None:rclpy.spin_once(n,timeout_sec=.1)
    if not state['injected']:raise RuntimeError('did not reach active final approach')
    if not state['result'] or not state['result'].startswith('FAILED:'):raise RuntimeError('loss did not fail the task')
    if state['press_seen']:raise RuntimeError('press phase entered after loss')
    if state.get('last_motion_after_injection',0)>1.5:raise RuntimeError('motion continued beyond watchdog')
    report={k:v for k,v in state.items() if k!='disable_future'}
    out=Path(__file__).resolve().parents[1]/'data/sam2_fov_validation/loss_probe.json'
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
finally:
    f=set_enabled(True);rclpy.spin_until_future_complete(n,f,timeout_sec=3)
    n.destroy_node();rclpy.shutdown()
