"""Read-only ROS snapshot for post-motion perception failure; sends no commands."""
import json,time
from pathlib import Path
from collections import Counter
import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from std_msgs.msg import String,Bool
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image,JointState,CameraInfo
from vision_msgs.msg import Detection2DArray
from rosidl_runtime_py.convert import message_to_ordereddict

rclpy.init();node=Node('read_only_approach_failure_snapshot')
root=Path(__file__).resolve().parents[1]/'data'/('approach_failure_'+time.strftime('%Y%m%d_%H%M%S'))
root.mkdir(parents=True)
qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
latched=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
counts=Counter();values={};bridge=CvBridge();images={}
def receive(topic,msg):
 counts[topic]+=1
 if isinstance(msg,Image):
  if topic not in images:
   im=bridge.imgmsg_to_cv2(msg,desired_encoding='passthrough')
   if msg.encoding=='rgb8':im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
   name='depth.png' if 'depth' in topic else 'color.png'
   cv2.imwrite(str(root/name),im);images[topic]=message_to_ordereddict(msg.header)
 elif isinstance(msg,String):
  try:values[topic]=json.loads(msg.data)
  except ValueError:values[topic]=msg.data
 else: values[topic]=message_to_ordereddict(msg)
for topic,typ,policy in [
 ('/button_approach_planner/observation_status',String,latched),
 ('/button_approach/status',String,latched),('/button_selected',String,latched),
 ('/button_tracking_state',String,qos),('/button_detection_valid',Bool,qos),
 ('/button_surface_pose',PoseStamped,qos),('/button_pose',PoseStamped,qos),
 ('/button_detections',Detection2DArray,qos),('/feedback/joint_states',JointState,qos),
 ('/control/joint_states',JointState,qos),('/camera/color/camera_info',CameraInfo,qos),
 ('/camera/color/image_raw',Image,qos),('/camera/aligned_depth_to_color/image_raw',Image,qos)]:
 node.create_subscription(typ,topic,lambda msg,t=topic:receive(t,msg),policy)
end=time.monotonic()+6
while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.1)
result={'counts':dict(counts),'values':values,'images':images,'nodes':node.get_node_names_and_namespaces()}
(root/'snapshot.json').write_text(json.dumps(result,indent=2))
print(root)
print(json.dumps({'counts':dict(counts),'selected':values.get('/button_selected'),'tracking':values.get('/button_tracking_state'),'observation':values.get('/button_approach_planner/observation_status')},indent=2))
node.destroy_node();rclpy.shutdown()
