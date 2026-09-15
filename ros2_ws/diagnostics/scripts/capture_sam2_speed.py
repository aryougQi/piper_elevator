#!/usr/bin/env python3
"""Read-only wall-time throughput and latency capture for a running simulation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import subprocess
import threading
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped, PointStamped, TwistStamped
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--seconds', type=float, default=180.)
p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'data/sam2_speed/live.json')
a=p.parse_args()
a.output.parent.mkdir(parents=True, exist_ok=True)
rclpy.init()
n=rclpy.create_node('sam2_speed_capture')
records=defaultdict(list); events=[]; clock=[]; performance=[]
start=time.monotonic(); done=None; task_started=False
receipts={}; ages=defaultdict(list); latencies=defaultdict(list); gpu=[]; stopped=threading.Event()
def sample_gpu():
    while not stopped.is_set():
        try:
            row=subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],text=True,timeout=2).strip()
            gpu.append([time.monotonic()-start,*map(float,row.split(","))])
        except (OSError,ValueError,subprocess.SubprocessError): pass
        stopped.wait(1.)
threading.Thread(target=sample_gpu,daemon=True).start()
q=QoSProfile(depth=50, durability=DurabilityPolicy.TRANSIENT_LOCAL)
def stream(topic,msg):
    stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
    now=time.monotonic()
    records[topic].append([now-start,stamp])
    if topic=='/camera/color/image_raw':
        receipts[stamp]=now
        if len(receipts)>300: del receipts[next(iter(receipts))]
    elif topic.startswith('/sam2_button_tracker/'):
        if clock: ages[topic].append((clock[-1][1]-stamp)*1000)
        if stamp in receipts: latencies[topic].append((now-receipts[stamp])*1000)
def event(topic,msg):
    global done,task_started
    events.append([time.monotonic()-start,topic,msg.data])
    if topic=='/elevator_task/result':
        if msg.data.startswith('RUNNING:'): task_started=True
        if task_started and msg.data.startswith(('COMPLETE:', 'FAILED:')): done=time.monotonic()
for topic,typ in [('/joint_states',JointState),('/servo_node/delta_twist_cmds',TwistStamped),('/camera/color/image_raw',Image),('/camera/aligned_depth_to_color/image_raw',Image),
                  ('/sam2_button_tracker/center',PointStamped),('/button_detections',Detection2DArray),('/sam2_button_tracker/surface_pose',PoseStamped),('/sam2_button_tracker/debug_image',Image)]:
    n.create_subscription(typ,topic,lambda m,t=topic:stream(t,m),qos_profile_sensor_data)
n.create_subscription(Clock,'/clock',lambda m:clock.append([time.monotonic()-start,m.clock.sec+m.clock.nanosec/1e9]),qos_profile_sensor_data)
for topic in ('/elevator_task/result','/elevator_task/status','/button_visual_servo/status','/button_approach/status','/button_tracking_state'):
    n.create_subscription(String,topic,lambda m,t=topic:event(t,m),q)
n.create_subscription(String,'/sam2_button_tracker/performance',lambda m:performance.append(json.loads(m.data)),10)
while time.monotonic()-start<a.seconds and (done is None or time.monotonic()-done<2):
    rclpy.spin_once(n,timeout_sec=.05)
def summarize(rows):
    v=np.array(rows)
    if len(v)<2:return {'count':len(v)}
    return dict(count=len(v),wall_fps=float((len(v)-1)/(v[-1,0]-v[0,0])),
                capture_fps=float((len(v)-1)/(v[-1,1]-v[0,1])),
                wall_gap_p95_ms=float(np.percentile(np.diff(v[:,0]),95)*1000))
stopped.set()
def distribution(v):
    return dict(mean=float(np.mean(v)),p95=float(np.percentile(v,95)),max=float(np.max(v))) if len(v) else None
report=dict(ages_ros_ms={k:distribution(v) for k,v in ages.items()},
            camera_receipt_to_target_ms={k:distribution(v) for k,v in latencies.items()},
            gpu_utilization=distribution([row[1] for row in gpu]),gpu_samples=gpu,
            stream_records=dict(records),duration_seconds=time.monotonic()-start,streams={k:summarize(v) for k,v in records.items()},
            rtf=(clock[-1][1]-clock[0][1])/(clock[-1][0]-clock[0][0]) if len(clock)>1 else None,
            performance=performance,events=events)
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('events','performance','stream_records','gpu_samples')},indent=2),flush=True)
n.destroy_node();rclpy.shutdown()
