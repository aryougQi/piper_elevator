#!/usr/bin/env python3
"""Capture actual camera projection and strict SAM2 handover in a simulation domain."""
import json
from pathlib import Path
import time
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from piper_elevator_app.motion_core import quaternion_to_matrix

class Capture(Node):
    def __init__(self):
        super().__init__('sam2_fov_capture')
        self.declare_parameter('use_sim_time', True) if not self.has_parameter('use_sim_time') else None
        self.image_at = 0.0
        self.bridge = CvBridge()
        self.phase = ''
        self.button = None
        self.info = None
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        out = Path(__file__).resolve().parents[1] / 'data/sam2_fov_validation'
        out.mkdir(parents=True, exist_ok=True)
        self.stream = (out / f'evidence_{time.time_ns()}.jsonl').open('w', buffering=1)
        latched = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for topic in ('/elevator_task/status', '/elevator_task/result', '/button_tracking_state', '/button_visual_servo/status', '/button_approach/status'):
            self.create_subscription(String, topic, lambda m, t=topic: self.event(t,m), latched)
        self.create_subscription(PoseStamped, '/button_pose_base', self.target, 10)
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera, 10)
        self.create_subscription(PoseStamped, '/sam2_button_tracker/surface_pose', lambda m: self.write('sam2_pose', stamp=m.header.stamp.sec+m.header.stamp.nanosec/1e9, position=[m.pose.position.x,m.pose.position.y,m.pose.position.z]),10)
        self.create_subscription(Image, '/sam2_button_tracker/debug_image', self.picture, 1)
        self.create_timer(0.03, self.sample)
    def picture(self, msg):
        if time.monotonic() - self.image_at < 2.0: return
        self.image_at = time.monotonic()
        target = self.stream.name.replace('.jsonl', '_frames')
        Path(target).mkdir(exist_ok=True)
        cv2.imwrite(str(Path(target)/f'{time.time_ns()}.jpg'), self.bridge.imgmsg_to_cv2(msg, 'bgr8'))

    def write(self, kind, **data):
        self.stream.write(json.dumps(dict(kind=kind, wall=time.time(), phase=self.phase, **data))+'\n')
    def event(self, topic, msg):
        if topic == '/elevator_task/status': self.phase=msg.data
        self.write(topic, data=msg.data)
    def target(self,msg):
        p=msg.pose.position
        self.button=np.array([p.x,p.y,p.z])
    def camera(self,msg): self.info=msg
    def sample(self):
        if self.button is None or self.info is None or not self.phase.startswith('COARSE_EXECUTING'): return
        try:
            t=self.tf.lookup_transform('base_link', self.info.header.frame_id, Time()).transform
            q=t.rotation; p=t.translation
            xyz=quaternion_to_matrix([q.x,q.y,q.z,q.w]).T @ (self.button-[p.x,p.y,p.z])
            u=self.info.k[0]*xyz[0]/xyz[2]+self.info.k[2]
            v=self.info.k[4]*xyz[1]/xyz[2]+self.info.k[5]
            margin=min(u,v,self.info.width-u,self.info.height-v)
            self.write('projection',u=float(u),v=float(v),z=float(xyz[2]),margin=float(margin))
        except Exception as exc: self.write('projection_error',error=str(exc))

rclpy.init()
node=Capture()
try: rclpy.spin(node)
finally:
    node.stream.close(); node.destroy_node(); rclpy.shutdown()
